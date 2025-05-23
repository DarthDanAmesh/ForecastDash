# deepar_model.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from constants import STANDARD_COLUMNS, DEFAULT_PREDICTION_LENGTH, MAX_ENCODER_LENGTH, BATCH_SIZE, MAX_EPOCHS
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import NormalDistributionLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepARModel:
    def __init__(self, df: pd.DataFrame, params: Optional[Dict] = None, max_epochs: int = MAX_EPOCHS):
        """Initialize DeepAR model with dataset and parameters."""
        self.df = df
        self.model = None
        self.training_dataset = None
        self.validation_dataset = None
        self.params = params or {}
        self.prediction_length = self.params.get('prediction_length', DEFAULT_PREDICTION_LENGTH)
        self.batch_size = self.params.get('batch_size', BATCH_SIZE)
        self.max_epochs = self.params.get('max_epochs', max_epochs)
        self.time_idx = 'time_idx'
        self.target = STANDARD_COLUMNS['demand']
        self.group_ids = [STANDARD_COLUMNS['material']]
        self.static_categoricals = [STANDARD_COLUMNS['material'], STANDARD_COLUMNS['country']]
        self.time_varying_known_reals = ['month', 'quarter', 'day_of_week', 'delivery_delay']
        self.time_varying_unknown_reals = [self.target]
        logger.info(f"Initialized DeepARModel with max_epochs={self.max_epochs}")

    def create_dataset(self) -> Optional[tuple[TimeSeriesDataSet, TimeSeriesDataSet]]:
        """Create training and validation TimeSeriesDataSets for DeepAR."""
        try:
            df = self.df.copy()
            required_cols = [self.time_idx, self.target] + self.group_ids
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns for DeepAR: {missing_cols}")
                st.error(f"Missing columns: {missing_cols}.", icon="🚨")
                return None

            # Ensure correct dtypes
            df[self.time_idx] = df[self.time_idx].astype('int64')
            df[self.target] = df[self.target].astype('float32')
            df[self.group_ids[0]] = df[self.group_ids[0]].astype(str)
            if 'delivery_delay' in df.columns:
                df['delivery_delay'] = df['delivery_delay'].astype('float32')

            # Filter groups with sufficient data
            min_length = MAX_ENCODER_LENGTH + self.prediction_length
            group_counts = df.groupby(self.group_ids[0])[self.time_idx].count()
            valid_groups = group_counts[group_counts >= min_length].index
            if len(valid_groups) == 0:
                logger.error(f"No groups have enough data points (minimum {min_length} required)")
                st.error(f"No groups have enough data points (minimum {min_length} required).", icon="🚨")
                return None
            df_filtered = df[df[self.group_ids[0]].isin(valid_groups)].sort_values([self.group_ids[0], self.time_idx])

            # Split into training and validation
            training_cutoff = df_filtered[self.time_idx].max() - self.prediction_length
            train_data = df_filtered[df_filtered[self.time_idx] <= training_cutoff]
            val_data = df_filtered[df_filtered[self.time_idx] > training_cutoff - MAX_ENCODER_LENGTH]

            if train_data.empty or val_data.empty:
                logger.error("Training or validation data is empty after splitting")
                st.error("Insufficient data for training and validation.", icon="🚨")
                return None

            # Create training dataset
            training_dataset = TimeSeriesDataSet(
                train_data,
                time_idx=self.time_idx,
                target=self.target,
                group_ids=self.group_ids,
                categorical_encoders={STANDARD_COLUMNS['material']: NaNLabelEncoder().fit(df[STANDARD_COLUMNS['material']])},
                min_encoder_length=2,
                max_encoder_length=MAX_ENCODER_LENGTH,
                max_prediction_length=self.prediction_length,
                static_categoricals=[c for c in self.static_categoricals if c in df.columns],
                time_varying_known_reals=[c for c in self.time_varying_known_reals if c in df.columns],
                time_varying_unknown_reals=self.time_varying_unknown_reals,
                allow_missing_timesteps=True,
                target_normalizer='auto'
            )

            # Create validation dataset
            validation_dataset = TimeSeriesDataSet.from_dataset(
                training_dataset,
                val_data,
                min_prediction_idx=training_cutoff + 1,
                stop_randomization=True
            )

            logger.info(f"DeepAR datasets created: {len(training_dataset)} training samples, {len(validation_dataset)} validation samples")
            return training_dataset, validation_dataset
        except Exception as e:
            logger.error(f"DeepAR dataset creation failed: {str(e)}", exc_info=True)
            st.error(f"DeepAR dataset creation failed: {str(e)}.", icon="🚨")
            return None

    def train(self) -> bool:
        """Train the DeepAR model."""
        try:
            datasets = self.create_dataset()
            if not datasets:
                logger.error("Failed to create datasets for training")
                return False
            self.training_dataset, self.validation_dataset = datasets

            train_dataloader = self.training_dataset.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)
            val_dataloader = self.validation_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

            self.model = DeepAR.from_dataset(
                self.training_dataset,
                learning_rate=0.01,
                hidden_size=16,
                rnn_layers=2,
                dropout=0.1,
                loss=NormalDistributionLoss(),
                log_interval=10,
                log_val_interval=1
            )

            trainer = Trainer(
                max_epochs=self.max_epochs,
                accelerator='auto',
                enable_progress_bar=False,
                logger=TensorBoardLogger(save_dir="lightning_logs", name="deepar"),
                callbacks=[EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, mode="min")]
            )

            trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            logger.info("DeepAR training completed")
            return True
        except Exception as e:
            logger.error(f"DeepAR training failed: {str(e)}", exc_info=True)
            st.error(f"DeepAR training failed: {str(e)}.", icon="🚨")
            self.model = None
            return False

    def predict(self, df: pd.DataFrame, periods: int) -> Optional[pd.DataFrame]:
        """Generate forecasts using DeepAR."""
        try:
            if not self.model or not self.training_dataset:
                logger.error("No trained DeepAR model or dataset available")
                st.error("No trained DeepAR model available.", icon="🚨")
                return None

            df = df.copy()
            max_date = df[STANDARD_COLUMNS['date']].max()
            future_dates = pd.date_range(
                start=max_date + pd.Timedelta(weeks=1),
                periods=periods,
                freq='W'
            )

            # Prepare prediction dataset
            pred_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset,
                df,
                predict=True,
                stop_randomization=True
            )
            pred_dataloader = pred_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
            predictions = self.model.predict(pred_dataloader, mode='raw', return_x=True)

            # Process predictions
            output = predictions.output
            group_ids = predictions.x['groups'].cpu().numpy()
            unique_group_ids = np.unique(group_ids)
            group_encoder = self.training_dataset.categorical_encoders[STANDARD_COLUMNS['material']]
            group_names = group_encoder.inverse_transform(unique_group_ids)

            forecast_list = []
            for group_id, group_name in zip(unique_group_ids, group_names):
                batch_idx = np.where(group_ids == group_id)[0][0]
                group_prediction = output.prediction[batch_idx]
                for t, date in enumerate(future_dates[:group_prediction.shape[0]]):
                    samples = group_prediction[t]
                    if np.isnan(samples).any():
                        logger.warning(f"NaN detected in predictions for {group_name} at step {t}")
                        continue
                    mean_forecast = samples.mean().item()
                    lower_bound = np.quantile(samples.numpy(), 0.1)
                    upper_bound = np.quantile(samples.numpy(), 0.9)
                    forecast_list.append({
                        'date': date,
                        STANDARD_COLUMNS['material']: str(group_name),
                        'forecast': mean_forecast,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    })

            if forecast_list:
                forecast_df = pd.DataFrame(forecast_list)
                logger.info("DeepAR predictions generated successfully")
                return forecast_df
            logger.warning("No forecasts generated")
            return None
        except Exception as e:
            logger.error(f"DeepAR prediction failed: {str(e)}", exc_info=True)
            st.error(f"DeepAR prediction failed: {str(e)}.", icon="🚨")
            return None
    #SHAP
    def predict_for_shap(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict method for SHAP KernelExplainer.
        X: DataFrame or numpy array of features (should match the columns used for training, including time_idx, group_ids, etc.)
        Returns: 1D numpy array of mean predictions for each row.
        """
        if not self.model or not self.training_dataset:
            raise RuntimeError("DeepAR model is not trained or dataset is missing.")

        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                columns = self.feature_names_
            else:
                columns = self.df.drop(columns=[self.target]).columns.tolist()
            X = pd.DataFrame(X, columns=columns)

        # Ensure the target column exists (fill with NaN if missing)
        if self.target not in X.columns:
            X[self.target] = np.nan

        # Ensure the time_idx column exists and is integer
        if self.time_idx in X.columns:
            X[self.time_idx] = X[self.time_idx].astype('int64')

        try:
            pred_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset,
                X,
                predict=True,
                stop_randomization=True
            )
            pred_dataloader = pred_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
            predictions = self.model.predict(pred_dataloader, mode='raw', return_x=True)
            output = predictions.output
            mean_preds = []
            for batch_idx in range(output.prediction.shape[0]):
                mean_pred = output.prediction[batch_idx, 0, :].mean().item()
                mean_preds.append(mean_pred)
            return np.array(mean_preds)
        except Exception as e:
            logger.error(f"predict_for_shap failed: {str(e)}", exc_info=True)
            raise