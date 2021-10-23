from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env.

TRAINING_FILE = "input/dri_wat_pot_folds.csv"
MODEL_OUTPUT = "local_models/" if os.getenv('SAVE_MODELS_IN_LOCAL',
                                            'False').lower() in ('true', '1', 't') else "models/"
