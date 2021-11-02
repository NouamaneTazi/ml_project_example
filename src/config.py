from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env.

# About files
RAW_FILE = "data/drinking_water_potability.csv"
TRAINING_FILE = "input/dri_wat_pot_folds.csv"
TESTING_FILE = "input/dri_wat_pot_test.csv"

# General parameters
RANDOM_STATE = 20
TEST_SIZE = 0.2
NUMBER_FOLDS = 5

# Save folders
SAVED_MODELS = "local_models/" if os.getenv('SAVE_MODELS_IN_LOCAL',
                                            'True').lower() in ('true', '1', 't') else "models/"
PREPROCESSING_PIPELINES = "local_pre_pipelines/" if os.getenv('SAVE_MODELS_IN_LOCAL',
                                                              'False').lower() in ('true', '1', 't') else "local_pre_pipelines/"
PREDICTIONS = "predictions/"
LOGS_FILE = "logs/scores.local.csv" if os.getenv('SAVE_SCORES_IN_LOCAL',
                                                 'True').lower() in ('true', '1', 't') else "logs/scores.csv"
