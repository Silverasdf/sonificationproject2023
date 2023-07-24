# Pytorch Lightning train, eval, test, save - Ryan Peruski, 06/21/2023
# Takes all the code from old train, eval, test, save and puts it into a Pytorch Lightning module
import os, sys
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import torch
import warnings
import importlib.util
warnings.filterwarnings("ignore", ".*does not have many workers.*")

#My modules
from mydms import ImageData
from mymodels import LitModelSave, ViTClassifierSave, LitModel, ViTClassifier
from cleanup import disperse_files, remove_directory
def init():

    # Example usage
    if len(sys.argv) < 2:
        print("usage: python pl_trainevaltestsave.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    CFG = config_module

    #Create model and save directories
    if not os.path.exists(CFG.MODEL_DIR):
        os.makedirs(CFG.MODEL_DIR)
    if not os.path.exists(CFG.SAVE_DIR):
        os.makedirs(CFG.SAVE_DIR)
    return CFG

def main():
    if torch.cuda.is_available():
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

    CFG = init()
    # Trains and/or tests NUM_OF_MODELS times
    for i in range(CFG.NUM_OF_MODELS):

        # Disperse files
        if CFG.REDISPERSE and i % CFG.REDISPERSE_FREQUENCY == 0:
            remove_directory(CFG.OLD_DIR)
            disperse_files(CFG.DISPERSE_LOCATION, CFG.OLD_DIR)

        # Init DataModule
        dm = ImageData(data_dir=CFG.OLD_DIR, batch_size=CFG.BATCH_SIZE, num_classes=20)
        dm_2023 = ImageData(data_dir=CFG.NEW_DIR, batch_size=CFG.BATCH_SIZE, num_classes=20)
        # Init model from datamodule's attributes
        if CFG.SAVE:
            if CFG.MODEL_TYPE != "ViT":
                model = LitModelSave( # This model inherits from LitModel and automatically saves perf curves. Use LitModel if you don't want to save perf curves
                                        lr=CFG.lr, 
                                        num_classes=dm.num_classes, 
                                        load=[not CFG.TRAIN, i], 
                                        model_dir=CFG.MODEL_DIR,
                                        output_dir=CFG.SAVE_DIR,
                                        model=CFG.MODEL_TYPE,
                                        mode=CFG.MODE,
                                        test_data_dir=CFG.NEW_DIR
                                    )
            else:
                model = ViTClassifierSave( # This model inherits from ViTClassifier and automatically saves perf curves. Use ViTClassifier if you don't want to save perf curves
                                        lr=CFG.lr, 
                                        num_classes=dm.num_classes, 
                                        load=[not CFG.TRAIN, i], 
                                        model_dir=CFG.MODEL_DIR,
                                        output_dir=CFG.SAVE_DIR,
                                        model=CFG.MODEL_TYPE,
                                        mode=CFG.MODE,
                                        test_data_dir=CFG.NEW_DIR
                                    )
        else:
            if CFG.MODEL_TYPE != "ViT":
                model = LitModel( # This model inherits from LitModel and automatically saves perf curves. Use LitModel if you don't want to save perf curves
                                        lr=CFG.lr, 
                                        num_classes=dm.num_classes, 
                                        load=[not CFG.TRAIN, i], 
                                        model_dir=CFG.MODEL_DIR,
                                        model=CFG.MODEL_TYPE,
                                    )
            else:
                model = ViTClassifier( # This model inherits from ViTClassifier and automatically saves perf curves. Use ViTClassifier if you don't want to save perf curves
                                        lr=CFG.lr, 
                                        num_classes=dm.num_classes, 
                                        load=[not CFG.TRAIN, i], 
                                        model_dir=CFG.MODEL_DIR,
                                        model=CFG.MODEL_TYPE,
                                    )
        # Init trainer
        trainer = L.Trainer(
            max_epochs=CFG.EPOCHS,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=[1],
            callbacks=[EarlyStopping(monitor='val_loss', patience=CFG.PATIENCE)],
        )
        # Train
        if CFG.TRAIN:
            trainer.fit(model, dm)

        #Save model
        if CFG.SAVE and CFG.TRAIN:
            model_num = i
            while os.path.exists(CFG.MODEL_DIR + '/new_model_' + str(model_num) + '.pth'):
                model_num += 1
            torch.save(model, CFG.MODEL_DIR + '/new_model_' + str(model_num) + '.pth')

        # Test
        #trainer.test(model, dm)
        trainer.test(model, dm_2023)


if __name__ == "__main__":
    main()