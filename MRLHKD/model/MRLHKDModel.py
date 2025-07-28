import pandas as pd
from MRLHKD.model.Net import Net
import torch
import numpy as np
from MRLHKD.utils.dataloader import make_dataloader
from MRLHKD.utils.logger import my_logger
from MRLHKD.utils.loss import CoxLoss, Knowledge_decomposition, survival_loss
from MRLHKD.utils.metric import calculate_c_index, calculate_mae
from MRLHKD.utils.utils import get_optimizer, EarlyStopping

class MRLHKDModel:
    duration_col = 'duration'
    event_col = 'event'
    def __init__(self, learning_rate=1e-3, dropout=0.2, output_size=64, hidden_dim=128, alpha=0.1, beta=0.1,
                 random_seed=2025, epochs=250, batch_size=1024, optimizer="Adam", val_size=0.2, device=torch.device("cuda")):
        self.baseline_hazards_ = None
        self.learning_rate = learning_rate
        self.dropout_rate = dropout
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.val_size = val_size
        self.device = device
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.fitted = False
        self.model = None

    def get_torch_model(self, clinicalDim, geneDim):
        return Net(clinical_dim=clinicalDim, gene_dim=geneDim, patch_num=40, wsi_dim=256, output_size=self.output_size,
                   hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate).to(self.device)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path,clinical_x_train, gene_x_train, WSIs_x_train, label_train):
        clinicalDim = clinical_x_train.shape[1]
        geneDim = gene_x_train.shape[1]
        self.model = self.get_torch_model(clinicalDim, geneDim)
        self.model.load_state_dict(torch.load(path))
        self.fitted = True
        # 计算基线风险
        self.compute_baseline_hazards(clinical_x_train, gene_x_train, WSIs_x_train, label_train)
        self.model.eval()


    def get_train_val_data(self, clinical_x, gene_x, WSIs_x, label):
        samples = clinical_x.shape[0]
        train_num = int(samples * (1 - self.val_size))
        all_index = np.array([i for i in range(samples)])
        np.random.seed(self.random_seed)
        np.random.shuffle(all_index)
        train_index = all_index[0:train_num]
        val_index = all_index[train_num:]
        clinical_x_train = clinical_x[train_index, :]
        clinical_x_val = clinical_x[val_index, :]
        gene_x_train = gene_x[train_index, :]
        gene_x_val = gene_x[val_index, :]
        WSIs_x_train = WSIs_x[train_index, :]
        WSIs_x_val = WSIs_x[val_index, :]
        label_train = label[train_index, :]
        label_val = label[val_index, :]
        return clinical_x_train, clinical_x_val, gene_x_train, gene_x_val, WSIs_x_train, \
            WSIs_x_val, label_train, label_val

    def fit(self, clinical_x, gene_x, WSIs_x, label):
        my_logger.info(f'hypeparameters: learning_rate {self.learning_rate}, output_size {self.output_size}, '
                       f'hidden_dim {self.hidden_dim}, alpha {self.alpha}, beta {self.beta}, '
                       f'dropout {self.dropout_rate}, '
                       f'batch_size {self.batch_size}, optimizer {self.optimizer}')
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        # data preprocess
        clinical_x_train, clinical_x_val, gene_x_train, gene_x_val, WSIs_x_train, \
            WSIs_x_val, label_train, label_val = self.get_train_val_data(clinical_x, gene_x, WSIs_x, label)
        clinicalDim = clinical_x.shape[1]
        geneDim = gene_x.shape[1]
        self.model = self.get_torch_model(clinicalDim, geneDim)
        optimizer = get_optimizer(self.model, self.learning_rate, self.optimizer)
        dics = []
        dics.append(self.model.state_dict())
        early_stopping = EarlyStopping(patience=5)
        for epoch in range(self.epochs):
            # train
            self.model.train()
            self.fitted = False
            my_logger.info('-------------epoch {}/{}-------------'.format(epoch + 1, self.epochs))
            train_loader = make_dataloader(clinical_x_train, gene_x_train, WSIs_x_train, label_train, self.batch_size)
            val_loader = make_dataloader(clinical_x_val, gene_x_val, WSIs_x_val, label_val, self.batch_size)
            total_train_loss = 0
            for _, (train_clinical_x, train_gene, train_WSIs, train_label) in enumerate(train_loader):
                tr_cspx,tr_gspx,tr_wspx,tr_cshx,tr_gshx,tr_wshx,tr_dualx1,tr_dualx2,tr_dualx3,tr_triple_share1,tr_triple_share2,tr_triple_share3,triple_share,tr_log_h,tr_log_st = self.model(
                    train_clinical_x.to(self.device), train_gene.to(self.device), train_WSIs.to(self.device))
                train_lcox = CoxLoss(tr_log_h, train_label.to(self.device), self.device)
                train_lsurv = survival_loss(tr_log_st, train_label.to(self.device))
                train_lkd = Knowledge_decomposition(tr_cspx,tr_gspx,tr_wspx,tr_cshx,tr_gshx,tr_wshx,tr_dualx1,tr_dualx2,tr_dualx3,tr_triple_share1,tr_triple_share2,tr_triple_share3,triple_share)
                train_loss = train_lcox + self.alpha * train_lsurv + self.beta * train_lkd
                # Backpropagation
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                total_train_loss = total_train_loss + train_loss.item()
                # assert not np.isnan(total_train_loss)
            total_train_loss = total_train_loss / len(train_loader)
            # validation
            self.model = self.model.eval()
            self.fitted = True
            # Calculate baseline hazard
            self.compute_baseline_hazards(clinical_x_train, gene_x_train, WSIs_x_train, label_train)
            train_cindex = self.get_c_index(clinical_x_train, gene_x_train, WSIs_x_train, label_train)
            val_cindex = self.get_c_index(clinical_x_val, gene_x_val, WSIs_x_val, label_val)
            train_mae = self.get_mae(clinical_x_train, gene_x_train, WSIs_x_train, label_train)
            val_mae = self.get_mae(clinical_x_val, gene_x_val, WSIs_x_val, label_val)
            total_val_loss = 0
            with torch.no_grad():
                for _, (val_clinical_x, val_gene, val_WSIs, val_label) in enumerate(val_loader):
                    val_cspx,val_gspx,val_wspx,val_cshx,val_gshx,val_wshx,val_dualx1,val_dualx2,val_dualx3,val_triple_share1,val_triple_share2,val_triple_share3,val_triple_share,val_log_h,val_log_st = self.model(
                        val_clinical_x.to(self.device), val_gene.to(self.device), val_WSIs.to(self.device))
                    val_lcox = CoxLoss(val_log_h, val_label.to(self.device), self.device)
                    val_lsurv = survival_loss(val_log_st, val_label.to(self.device))
                    val_lkd= Knowledge_decomposition(val_cspx,val_gspx,val_wspx,val_cshx,val_gshx,val_wshx,val_dualx1,val_dualx2,val_dualx3,val_triple_share1,val_triple_share2,val_triple_share3,val_triple_share)

                    val_loss = val_lcox + self.alpha * val_lsurv + self.beta * val_lkd
                    total_val_loss = total_val_loss + val_loss.item()
                total_val_loss = total_val_loss / len(val_loader)
            my_logger.info(f'train loss={format(total_train_loss, ".4f")}, '
                           f'val loss={format(total_val_loss, ".4f")}, '
                           f'train_cindex={format(train_cindex, ".4f")} '
                           f'train_mae={format(train_mae, ".4f")}, val_cindex={format(val_cindex, ".4f")},'
                           f' val_mae={format(val_mae, ".4f")},'
                           f'')

            # early stop
            early_stopping(total_val_loss, self.model, dics)
            if early_stopping.early_stop:
                my_logger.info("Early stopping")
                self.model.load_state_dict(dics[0])
                del dics
                break
        self.model = self.model.eval()
        self.fitted = True
        return self

    def target_to_df(self, target):
        durations, events = target[:, 0], target[:, 1]
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events})
        return df

    def compute_baseline_hazards(self, clinical_x, gene_x, WSIs_x, target=None, max_duration=None, sample=None, batch_size=8224,
                                 set_hazards=True, eval_=True, num_workers=0):
        df = self.target_to_df(target)  # .sort_values(self.duration_col)
        if sample is not None:
            if sample >= 1:
                df = df.sample(n=sample)
            else:
                df = df.sample(frac=sample)
        base_haz = self._compute_baseline_hazards(clinical_x, gene_x, WSIs_x, df, max_duration, batch_size,
                                                  eval_=eval_, num_workers=num_workers)
        if set_hazards:
            self.compute_baseline_cumulative_hazards(clinical_x, gene_x, WSIs_x, target=target, set_hazards=True, baseline_hazards_=base_haz)
        return base_haz

    def compute_baseline_cumulative_hazards(self, clinical_x, gene_x, WSIs_x, target=None, max_duration=None, sample=None,
                                            batch_size=8224, set_hazards=True, baseline_hazards_=None,
                                            eval_=True, num_workers=0):
        if baseline_hazards_ is None:
            baseline_hazards_ = self.compute_baseline_hazards(clinical_x, gene_x, WSIs_x, target, max_duration, sample, batch_size,
                                                              set_hazards=False, eval_=eval_, num_workers=num_workers)
        assert baseline_hazards_.index.is_monotonic_increasing, \
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        bch = (baseline_hazards_
               .cumsum()
               .rename('baseline_cumulative_hazards'))
        if set_hazards:
            self.baseline_hazards_ = baseline_hazards_
            self.baseline_cumulative_hazards_ = bch
        return bch

    def predict_cumulative_hazards(self, clinical_x, gene_x, WSIs_x, max_duration=None, batch_size=8224, verbose=False,
                                   baseline_hazards_=None, eval_=True, num_workers=0):
        """See `predict_survival_function`."""
        if baseline_hazards_ is None:
            if not hasattr(self, 'baseline_hazards_'):
                raise ValueError('Need to compute baseline_hazards_. E.g run `model.compute_baseline_hazards()`')
            baseline_hazards_ = self.baseline_hazards_
        assert baseline_hazards_.index.is_monotonic_increasing, \
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        return self._predict_cumulative_hazards(clinical_x, gene_x, WSIs_x, max_duration, batch_size, verbose, baseline_hazards_,
                                                eval_, num_workers=num_workers)

    def _compute_baseline_hazards(self, clinical_x, gene_x, WSIs_x, df_target, max_duration, batch_size, eval_=True, num_workers=0):
        if max_duration is None:
            max_duration = np.inf

        # Here we are computing when expg when there are no events.
        #   Could be made faster, by only computing when there are events.
        return (df_target
                .assign(expg=np.exp(self.predict(clinical_x, gene_x, WSIs_x)))
                .groupby(self.duration_col)
                .agg({'expg': 'sum', self.event_col: 'sum'})
                .sort_index(ascending=False)
                .assign(expg=lambda x: x['expg'].cumsum())
                .pipe(lambda x: x[self.event_col] / x['expg'])
                .fillna(0.)
                .iloc[::-1]
                .loc[lambda x: x.index <= max_duration]
                .rename('baseline_hazards'))

    def _predict_cumulative_hazards(self, clinical_x, gene_x, WSIs_x, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_=True, num_workers=0):
        max_duration = np.inf if max_duration is None else max_duration
        if baseline_hazards_ is self.baseline_hazards_:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(set_hazards=False,
                                                           baseline_hazards_=baseline_hazards_)
        bch = bch.loc[lambda x: x.index <= max_duration]
        expg = np.exp(self.predict(clinical_x, gene_x, WSIs_x)).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg),
                            index=bch.index)

    def predict_surv_df(self, clinical_x, gene_x, WSIs_x, max_duration=None, batch_size=8224, verbose=False, baseline_hazards_=None,
                        eval_=True, num_workers=0):
        """Predict survival function for `input`. S(x, t) = exp(-H(x, t))
        Require computed baseline hazards.

        Returns:
            pd.DataFrame -- Survival estimates. One columns for each individual.
        """
        return np.exp(-self.predict_cumulative_hazards(clinical_x, gene_x, WSIs_x, max_duration, batch_size, verbose, baseline_hazards_,
                                                       eval_, num_workers))

    def predict(self, clinical_x, gene_x, WSIs_x):
        clinical_x = torch.from_numpy(clinical_x).float().to(self.device)
        gene_x = torch.from_numpy(gene_x).float().to(self.device)
        WSIs_x = torch.from_numpy(WSIs_x).float().to(self.device)
        if self.fitted:
            with torch.no_grad():
                cspx,gspx,wspx,cshx,gshx,wshx,dualx1,dualx2,dualx3,triple_share1,triple_share2,triple_share3,triple_share,log_h,log_st = self.model(clinical_x, gene_x, WSIs_x)
                log_h = log_h.cpu().numpy()
                return log_h
        else:
            raise Exception("The model has not been fitted yet.")

    def getSurvivalTime(self, clinical_x, gene_x, WSIs_x):
        clinical_x = torch.from_numpy(clinical_x).float().to(self.device)
        gene_x = torch.from_numpy(gene_x).float().to(self.device)
        WSIs_x = torch.from_numpy(WSIs_x).float().to(self.device)
        if self.fitted:
            with torch.no_grad():
                cspx,gspx,wspx,cshx,gshx,wshx,dualx1,dualx2,dualx3,triple_share1,triple_share2,triple_share3,triple_share,log_h,log_st = self.model(clinical_x, gene_x, WSIs_x)
                log_st = log_st.exp().cpu().numpy()
                return log_st
        else:
            raise Exception("The model has not been fitted yet.")

    def get_c_index(self, clinical_x, gene_x, WSIs_x, label):
        surv = self.predict_surv_df(clinical_x, gene_x, WSIs_x)
        time_index, probability_array = np.array(surv.index), surv.values
        return calculate_c_index(time_index, probability_array, label)

    def get_mae(self, clinical_x, gene_x, WSIs_x, label):
        predict_time = self.getSurvivalTime(clinical_x, gene_x, WSIs_x)
        return calculate_mae(predict_time, label)

