# Cliptrain from "Simple Implemtation" - Ryan Peruski, 07/01/2023
# This takes from the openai CLIP github and modifies it to work with the blur image dataset
import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
import albumentations as A
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import warnings
from PIL import Image
import torch
import torchvision.transforms as transforms
from PIL import Image
import clip
import sys
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

# Include argument for front or back
if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    print('usage: python openai_clip_simple_implementation.py -front/-back')
    sys.exit(1)

#Backseat
if mode == '-back':
    image_path = '/root/BlurImageTrainingProject/Data_Back/GroundTruth/1'
    image_path2 = '/root/BlurImageTrainingProject/Data_Back/GroundTruth/0'
    test_image_path = '/root/BlurImageTrainingProject/Data_Back/New_Data_2023_edit/Testing/1'
    test_image_path2 = '/root/BlurImageTrainingProject/Data_Back/New_Data_2023_edit/Testing/0'
    model_path = '/root/BlurImageTrainingProject/Experiments/CLIP_Back_Models'
    result_dir = '/root/BlurImageTrainingProject/Experiments/CLIP_Back_Results'
#Frontseat
elif mode == '-front':
    image_path = '/root/BlurImageTrainingProject/Data_Front/TrainingAndValidation/1'
    image_path2 = '/root/BlurImageTrainingProject/Data_Front/TrainingAndValidation/0'
    test_image_path = '/root/BlurImageTrainingProject/Data_Front/New_Data_2023_edit/Testing/1'
    test_image_path2 = '/root/BlurImageTrainingProject/Data_Front/New_Data_2023_edit/Testing/0'
    model_path = '/root/BlurImageTrainingProject/Experiments/CLIP_Front_Models'
    result_dir = '/root/BlurImageTrainingProject/Experiments/CLIP_Front_Results'
else:
    print('usage: python openai_clip_simple_implementation.py -front/-back')
    sys.exit(1)
df = pd.DataFrame(columns=['image', 'caption', 'id'])
dict_ = {'image': [], 'caption': []}
for file in os.listdir(image_path):
    if not file.endswith('.jpg'):
        continue
    dict_['image'].append(file)
    dict_['caption'].append("A picture of a person")
for file in os.listdir(image_path2):
    if not file.endswith('.jpg'):
        continue
    dict_['image'].append(file)
    dict_['caption'].append("A picture of an empty seat")
df['image'] = dict_['image']
df['caption'] = dict_['caption']
df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]

df_test = pd.DataFrame(columns=['image', 'caption', 'id'])
dict_ = {'image': [], 'caption': []}
for file in os.listdir(test_image_path):
    if not file.endswith('.jpg'):
        continue
    dict_['image'].append(file)
    dict_['caption'].append("A picture of a person")
for file in os.listdir(test_image_path2):
    if not file.endswith('.jpg'):
        continue
    dict_['image'].append(file)
    dict_['caption'].append("A picture of an empty seat")
df_test['image'] = dict_['image']
df_test['caption'] = dict_['caption']
df_test['id'] = [id_ for id_ in range(df_test.shape[0] // 5) for _ in range(5)]

"""## Config

*A note on config and CFG: I wrote the codes with python scripts and then converted it into a Jupyter Notebook. So, in case of python scripts, config is a normal python file where I put all the hyperparameters and in the case of Jupyter Notebook, its a class defined in the beginning of the notebook to keep all the hyperparameters.*
"""

class CFG:
    debug = False
    result_dir = result_dir
    image_path = image_path
    image_path2 = image_path2
    image_path_test = test_image_path
    image_path_test2 = test_image_path2
    model_path = model_path
    captions = df
    captions_test = df_test
    batch_size = 4
    num_workers = 1
    head_lr = 1e-5
    num_models=10
    image_encoder_lr = 1e-5
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 10
    factor = 0.8
    epochs = 1000
    train = False
    save = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model_name = 'resnet50'
    model_name = 'vit_base_patch16_224'
    image_embedding = 768 #2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1

"""## Utils"""

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

"""## Dataset

As you can see in the tittle image of this article, we need to encode both images and their describing texts. So, the dataset needs to **return both images and texts**. Of course we are not going to feed raw text to our text encoder! We will use **DistilBERT** model (which is smaller than BERT but performs nearly as well as BERT) from **HuggingFace** library as our text encoder; so, we need to **tokenize** the sentences (captions) with DistilBERT tokenizer and then feed the token ids (input_ids) and the attention masks to DistilBERT. Therefore, the dataset needs to take care of the tokenization as well. Below you can see the dataset's code. Below that I'll explain the most important things that is happening in the code.

In the **\_\_init\_\_** we receive a tokenizer object which is actually a HuggingFace tokinzer; this tokenizer will be loaded when running the model. We are padding and truncating the captions to a specified max_length. In the **\_\_getitem\_\_** we will first load an encoded caption which is a dictionary with keys input_ids and attention_mask, make tensors out of its values and after that we will load the corresponding image, transform and augment it (if there is any!) and then we make it a tensor and put it in the dictionary with "image" as the key. Finally we put the raw text of the caption with the key "caption" in the dictionary only for visualization purposes.

I did not use additional data augmentations but you can add them if you want to improve the model's performance.
"""

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        #Sorry for bad programming
        if os.path.exists(f"{CFG.image_path}/{self.image_filenames[idx]}"):
            image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        elif os.path.exists(f"{CFG.image_path2}/{self.image_filenames[idx]}"):
            image = cv2.imread(f"{CFG.image_path2}/{self.image_filenames[idx]}")
        elif os.path.exists(f"{CFG.image_path_test}/{self.image_filenames[idx]}"):
            image = cv2.imread(f"{CFG.image_path_test}/{self.image_filenames[idx]}")
        else:
            image = cv2.imread(f"{CFG.image_path_test2}/{self.image_filenames[idx]}")
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.CenterCrop(224, 224, always_apply=True),
                A.Flip(),
                A.Affine(rotate=0, translate_percent=(0.025, 0.025)),
                A.ColorJitter(brightness=0.2, contrast=0.2),
                A.Normalize(max_pixel_value=255.0, always_apply=True)
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.CenterCrop(224, 224, always_apply=True),
                A.Flip(),
                A.Affine(rotate=0, translate_percent=(0.025, 0.025)),
                A.ColorJitter(brightness=0.2, contrast=0.2),
                A.Normalize(max_pixel_value=255.0, always_apply=True)
            ]
        )

"""## Image Encoder

The image encoder code is straight forward. I'm using PyTorch Image Models library (timm) here which makes a lot of different image models available from ResNets to EfficientNets and many more. Here we will use a ResNet50 as our image encoder. You can easily use torchvision library to use ResNets if you don't want to install a new library.

The code encodes each image to a fixed size vector with the size of the model's output channels (in case of ResNet50 the vector size will be **2048**). This is the output after the nn.AdaptiveAvgPool2d() layer.
"""

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

"""## Text Encoder

As I mentioned before, I'll use DistilBERT as the text encoder. Like its bigger brother BERT, two special tokens will be added to the actual input tokens: **CLS** and **SEP** which mark the start and end of a sentence. To grab the whole representation of a sentence (as the related BERT and DistilBERT papers point out) we use the final representations of the CLS token and we hope that this representation captures the overall meaning of the sentence (caption). Thinking it in this way, it is similar to what we did to images and converted them into a fixed size vector.

In the case of DistilBERT (and also BERT) the output hidden representation for each token is a vector with size **768**. So, the whole caption will be encoded in the CLS token representation whose size is 768.
"""

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

"""## Projection Head

I used [Keras code example implementation](https://keras.io/examples/nlp/nl_image_search/) of projection head to write the following in PyTorch.
Now that we have encoded both our images and texts into fixed size vectors (2048 for image and 768 for text) we need to bring (project) them into a **new world** (!) with **similar dimensions** for both images and texts in order to be able to compare them and push apart the non-relevant image and texts and pull together those that match. So, the following code will bring the 2048 and 768 dimensional vectors into a 256 (projection_dim) dimensional world, where we can **compare** them.

"embedding_dim" is the size of the input vector (2048 for images and 768 for texts) and "projection_dim" is the the size of the output vector which will be 256 for our case. For understanding the details of this part you can refer to the CLIP paper.
"""

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

"""## CLIP

This part is where all the fun happens! I'll also talk about the loss function here. I translated some of the code from Keras code examples into PyTorch for writing this part. Take a look at the code and then read the explanation below this code block.

Here we will use the previous modules that we built to implement the main model. The \_\_init\_\_ function is self-explanatory. In the forward function, we first encode the images and texts separately into fixed size vectors (with different dimensionalities). After that, using separate projection modules we project them to that shared world (space) that I talked about previously. Here the encodings will become of similar shape (256 in our case). After that we will compute the loss. Again I recommend reading CLIP paper to get it better but I'll try my best to explain this part.

In **Linear Algebra**, one common way to measure if two vectors are of similar characteristics (they are like each other) is to calculate their **dot product** (multiplying the matching entries and take the sum of them); if the final number is big, they are alike and if it is small they are not (relatively speaking)!

Okay! What I just said is the most important thing to have in mind to understand this loss function. Let's continue. We talked about two vectors, but, what do we have here? We have image_embeddings, a matrix with shape (batch_size, 256) and text_embeddings with shape (batch_size, 256). Easy enough! it means we have two groups of vectors instead of two single vectors. How do we measure how similar two groups of vectors (two matrices) are to each other? Again, with dot product (@ operator in PyTorch does the dot product or matrix multiplication in this case). To be able to multiply these two matrices together, we transpose the second one. Okay, we get a matrix with shape (batch_size, batch_size) which we will call logits. (temperature is equal to 1.0 in our case, so, it does not make a difference. You can play with it and see what difference it makes. Also look at the paper to see why it is here!).

I hope you are still with me! If not it's okay, just review the code and check their shapes. Now that we have our logits, we need targets. I need to say that there is a more straight forward way to obtain targets but I had to do this for our case (I'll talk about why in a next paragraph).

Let's consider what we hope that this model learns: **we want it to learn "similar representations (vectors)" for a given image and the caption describing it. Meaning that either we give it an image or the text describing it, we want it to produce same 256 sized vectors for both.**

#### Check the cell below this code block for the continue of the explanations
"""

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    def get_sim(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        return logits



def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

"""So, in the best case scenario, text_embeddings and image_embedding matricies should be the same because they are describing similar things. Let's think now: if this happens, what would the logits matrix be like? Let's see with a simple example!"""

# A simple Example

batch_size = 4
dim = 256
embeddings = torch.randn(batch_size, dim)
out = embeddings @ embeddings.T
"""So logits, in the best case, will be a matrix that if we take its softmax, will have 1.0s in the diagonal (An identity matrix to call it with fancy words!). As the loss function's job is to make model's predictions similar to targets (at least in most cases!), we want such a matrix as our target. That's the reason why we are calculating images_similarity and texts_similarity matrices in the code block above.

Now that we've got our targets matrix, we will use simple cross entropy to calculate the actual loss. I've written the full matrix form of cross entropy as a function which you can see in the bottom of the code block. Okay! We are done! Wasn't it simple?! Alright, you can ignore the next paragraph but if you are curious, there is an important note in that.

**Here's why I didn't use a simpler approach**: I need to admit that there's a simpler way to calculate this loss in PyTorch; by doing this: nn.CrossEntropyLoss()(logits, torch.arange(batch_size)). Why I did not use it here? For 2 reasons. 1- The dataset we are using has multiple captions for a single image; so, there is the possibility that two identical images with their similar captions exist in a batch (it is rare but it can happen). Taking the loss with this easier method will ignore this possibility and the model learns to pull apart two representations (assume them different)  that are actually the same. Obviously, we don't want this to happen so I calculated the whole target matrix in a way that takes care of these edge cases. 2- Doing it the way I did, gave me a better understanding of what is happening in this loss function; so, I thought it would give you a better intuition as well!

## Train

Here are some funtions to help us load train and valid dataloaders, our model and then train and evaluate our model on those. There's not much going on here; just simple training loop and utility functions
"""

def make_train_valid_dfs(caption=CFG.captions, valid_value=0.2):
    dataframe = caption.copy()
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(valid_value * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

"""Here's a handy function to train our model. There's not much happening here; just loading the batches, feeding them to the model and stepping the optimizer and lr_scheduler."""

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def test_epoch(model, test_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model.get_sim(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def main(model_num=0):
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model, os.path.join(CFG.model_path, f"new_model_{model_num}.pth"))
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    model = torch.load(model_path).to(CFG.device)
    model.eval()

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

def find_matches(model, image_embeddings, query, image_filenames, num, n=9, maxi=0):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    labels = []
    preds = []
    num_ones =0
    for file in image_filenames:
        if os.path.exists(f'{image_path}/{file}') or os.path.exists(f'{test_image_path}/{file}'):
            labels.append(1)
            num_ones += 1
        else:
            labels.append(0)
    scores = [score[0] for score in dot_similarity.T.detach().cpu().numpy()]
    if maxi == 0:
        maxi = max(scores)
    new_scores = [score/maxi for score in scores]
    #Get best f1 score
    precision, recall, thres = precision_recall_curve(labels, new_scores)
    #List of f1 scores for each threshold
    f1_scores = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) for i in range(len(precision))]
    best_thresh = np.array(thres)[np.nanargmax(f1_scores)]
    best_f1 = max(f1_scores)
    print("Best F1: ", best_f1, "Best Threshold: ", best_thresh)
    #Normalize scores to be between 0 and 1
    for score in new_scores:
        if score >= best_thresh:
            preds.append(1)
        else:
            preds.append(0)
    print("Accuracy: ", accuracy_score(labels, preds))
    if CFG.save:
        df = pd.DataFrame({'name': image_filenames, 'y_true': labels, 'y_pred': preds, 'y_scores': new_scores, 'mode': "CLIP", 'prevalence': num_ones/len(image_filenames)})
        df.to_json(os.path.join(CFG.result_dir, f'CLIP_{num}_perfs.json'))

    # values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    # matches = [image_filenames[idx] for idx in indices[::5]]

    # _, axes = plt.subplots(3, 3, figsize=(10, 10))
    # for match, ax in zip(matches, axes.flatten()):
    #     if os.path.exists(f"{CFG.image_path}/{match}"):
    #         image = cv2.imread(f"{CFG.image_path}/{match}")
    #     else:
    #         image = cv2.imread(f"{CFG.image_path2}/{match}")
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     ax.imshow(image)
    #     ax.axis("off")

    # plt.show()
    return maxi

for i in range(CFG.num_models):
    if not CFG.train and not os.path.exists(os.path.join(CFG.model_path, f"new_model_{i}.pth")):
        print(f"Model {i} does not exist")
        continue
    if CFG.train:
        print(f"Training Model {i}")
        main(i)
    model_path = os.path.join(CFG.model_path, f"new_model_{i}.pth")
    _, valid_df = make_train_valid_dfs(caption=CFG.captions, valid_value=0.15)
    model, image_embeddings = get_image_embeddings(valid_df, model_path=model_path)
    maxi = find_matches(model,
             image_embeddings,
             query="a picture of a person",
             image_filenames=valid_df['image'].values,
             num=i,
             n=9)
    _, valid_df = make_train_valid_dfs(caption=CFG.captions_test,valid_value=1)
    model, image_embeddings = get_image_embeddings(valid_df, model_path=model_path)
    find_matches(model,
             image_embeddings,
             query="a picture of a person",
             image_filenames=valid_df['image'].values,
             num=i,
             n=9,
             maxi=maxi)

"""## Final words

I hope you have enjoyed this article. Implementing this paper was a really interesting experience for me. I want to thank Khalid Salama for the great Keras code example he provided which inspired me to write something similar in PyTorch.
"""