
from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import (
    read_yaml_file,
    write_yaml_file
)

from scipy.stats import ks_2samp
import pandas as pd
import os
import sys


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(
        self, dataframe: pd.DataFrame
    ) -> bool:

        try:
            schema_columns = self._schema_config.get(
                "columns", self._schema_config
            )

            number_of_columns = len(schema_columns)

            logging.info(
                f"Required number of columns: {number_of_columns}"
            )

            logging.info(
                f"Dataframe has columns: {len(dataframe.columns)}"
            )

            return len(dataframe.columns) == number_of_columns

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(
        self, base_df, current_df, threshold=0.05
    ) -> bool:

        try:
            status = True
            report = {}

            for column in base_df.columns:

                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dist = ks_2samp(d1, d2)

                if is_same_dist.pvalue >= threshold:
                    is_found = False
                else:
                    is_found = True
                    status = False

                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status": is_found
                    }
                })

            drift_report_file_path = (
                self.data_validation_config.drift_report_file_path
            )

            dir_path = os.path.dirname(drift_report_file_path)

            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(
                file_path=drift_report_file_path,
                content=report
            )

            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:

        try:
            train_file_path = (
                self.data_ingestion_artifact.trained_file_path
            )

            test_file_path = (
                self.data_ingestion_artifact.test_file_path
            )

            # Read train and test datasets

            train_dataframe = DataValidation.read_data(
                train_file_path
            )

            test_dataframe = DataValidation.read_data(
                test_file_path
            )

            # Validate number of columns

            if not self.validate_number_of_columns(train_dataframe):
                raise ValueError(
                    "Train dataframe does not contain all columns."
                )

            if not self.validate_number_of_columns(test_dataframe):
                raise ValueError(
                    "Test dataframe does not contain all columns."
                )

            # Validate matching column names

            if set(train_dataframe.columns) != set(test_dataframe.columns):
                raise ValueError(
                    "Train and test dataframe columns do not match."
                )

            # Detect dataset drift

            status = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe
            )

            # Initialize output paths

            valid_train_file_path = None
            valid_test_file_path = None
            invalid_train_file_path = None
            invalid_test_file_path = None

            if status:

                valid_train_file_path = (
                    self.data_validation_config.valid_train_file_path
                )

                valid_test_file_path = (
                    self.data_validation_config.valid_test_file_path
                )

                os.makedirs(
                    os.path.dirname(valid_train_file_path),
                    exist_ok=True
                )

                os.makedirs(
                    os.path.dirname(valid_test_file_path),
                    exist_ok=True
                )

                train_dataframe.to_csv(
                    valid_train_file_path,
                    index=False,
                    header=True
                )

                test_dataframe.to_csv(
                    valid_test_file_path,
                    index=False,
                    header=True
                )

            else:

                invalid_train_file_path = (
                    self.data_validation_config.invalid_train_file_path
                )

                invalid_test_file_path = (
                    self.data_validation_config.invalid_test_file_path
                )

                os.makedirs(
                    os.path.dirname(invalid_train_file_path),
                    exist_ok=True
                )

                os.makedirs(
                    os.path.dirname(invalid_test_file_path),
                    exist_ok=True
                )

                train_dataframe.to_csv(
                    invalid_train_file_path,
                    index=False,
                    header=True
                )

                test_dataframe.to_csv(
                    invalid_test_file_path,
                    index=False,
                    header=True
                )

            # Create data validation artifact

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=valid_train_file_path,
                valid_test_file_path=valid_test_file_path,
                invalid_train_file_path=invalid_train_file_path,
                invalid_test_file_path=invalid_test_file_path,
                drift_report_file_path=(
                    self.data_validation_config.drift_report_file_path
                )
            )

            logging.info(
                f"Data validation artifact: {data_validation_artifact}"
            )

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

