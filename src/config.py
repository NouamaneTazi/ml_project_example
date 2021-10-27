from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env.

NUMBER_FOLDS = 5
TRAINING_FILE = "input/dri_wat_pot_folds.csv"
SAVED_MODELS = "local_models/" if os.getenv('SAVE_MODELS_IN_LOCAL',
                                            'True').lower() in ('true', '1', 't') else "models/"
PREPROCESSING_PIPELINES = "local_pre_pipelines/" if os.getenv('SAVE_MODELS_IN_LOCAL',
                                                              'False').lower() in ('true', '1', 't') else "local_pre_pipelines/"
TESTING_FILE = "input/dri_wat_pot_test.csv"
PREDICTIONS = "predictions/"
LOGS_FILE = "logs/scores.local.csv" if os.getenv('SAVE_SCORES_IN_LOCAL',
                                                 'True').lower() in ('true', '1', 't') else "logs/scores.csv"
