# Cloud Classifier

Summary: An image classifier and inference deployment for classifying cloud type from the 10 main types. The 
    neural network is pre-trained with resnet34 to speed up the learning process. The 
    cloud_learner uses a collection of images downloaded from the web to train/refine the neural network. 
    The 10 main cloud types used here are:
    
    Cirrus
    Cirrocumulus
    Cirrostratus
    Altocumulus
    Altostatus
    Nimbostratus
    Stratocumulus
    Stratus
    Cumulus
    Cumulonimbus

# Table of Contents
1. [Introduction](#introduction)
2. [Setup Data with Labels](#setup-data-with-labels)
3. [Loading and Training](#loading-and-training)
4. [Clean the Data](#clean-the-data)
5. [Export the Model](#export-the-model)
6. [Conclusion](#conclusion)


## Introduction

Some clouds bring precipitation, some just make the sky look interesting or dull. Clouds might all look fairly similar to an untrainer eye but here I ask if there are differences that a neural network can pick out? In this project I train a model to pick out these differences from photos. The wrinkle in this project is that a lot of photo of the sky have limited colour range and may or may not have ground objects that give context.

The definitions of the cloud types are taken from the [Met Office](https://www.metoffice.gov.uk/weather/learn-about/weather/types-of-weather/clouds/cloud-names-classifications). In general, the name gives a clue about the hight and texture of the cloud. For the height, "cirro" are high cloud-base clouds (>6000 m), "alto" and "nimbostratus" are intermediate height cloud-base clouds (2000 - 6000 m), and the rest have a low cloud-base (<2000 m). Then for the texture, "cumulus" indicates fluffy, cauliflower like cloud, "status" are smooth blanket-like cloud, "nimbo" tent to be rain clouds that are thicker and therefore darker, and "cirrus" are light, spindly cloud.

While there is some logic in the naming, this is a lot of information to remember so it seems useful to be able to identify cloud type from a quick photo of the sky. Given that the cloud types have different texture and atmospheric heights a neural network may be able to learn features and give a reasonably accurate prediction of its type.

## Setup Data with Labels

I have collected photos from several sources. Many of these come from Microsoft Azure's Bing photo search. This is a tool that has limited free access and allowed me to start labelling photos based on searches. These initial searches were supplemented with photos from [Unsplash](https://unsplash.com/) and my own photos. Supplementing with my own photos does have a drawback of biassing the model towards where I live but was necessary to improve a small dataset. The bias is that cloud types can be more or less common in different parts of the world and have different atmospheric heights.


```python
# Load my Microsoft Azure key from file.
with open(key_file, 'r') as f:
  lines = f.readlines()
key = os.environ.get('AZURE_SEARCH_KEY', lines[0])


cloud_types = 'Cirrus', 'Cirrocumulus', 'Cirrostratus', 'Altocumulus', 'Altostratus', 'Nimbostratus', 'Stratocumulus', 'Stratus', 'Cumulus', 'Cumulonimbus'
cloud_height = '> 6000 m', '> 6000 m', '> 6000 m', '2000 - 6000 m', '2000 - 6000 m', '2000 - 6000 m', '< 2000 m', '< 2000 m', '< 2000 m', '< 2000 m'
path = Path('gdrive/MyDrive/Data_for_apps/Cloud_Learner/clouds')

if not path.exists():
    path.mkdir()
    for o in cloud_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} clouds')
        download_images(dest, urls=results.attrgot('contentUrl'))

```

I discovered some cloud types are return fewer results in searches than others. I think this is because some clouds are less interesting and photogenic than others biassing search results towards these. I found it harder to find photos of the smooth, blanket stratus at each atmospheric height and rain clouds. The final dataset had at least 90 photos of each type and around 1000 photos in total. This is a small dataset and made me realise how tricky it can be to get quality labelled training data. Let's see what we can do with what we've got.


```python
def get_equal_images(path, folders=((),), verbose=False):
  """ 
  A function to shuffle the list of files in each category for the data loader. 
  The function then makes sure there are the same number of photos in each category to reduce the bias
  towards the categories with more photos.
  This can also output a list of the number of files in each category.
  
  Parameters:
      path (string): Path to the data folders.
      folders (list): Folders to loop over in the path.
      verbose (bool): If True print the category with number of items.
  Returns:
      fns (list): A list of randomised photo paths.
  """

  fns = []
  min_type = 1e20
  for o in folders:
    fns_tmp = get_image_files(path, folders=o)
    if len(fns_tmp) < min_type:
      min_type = len(fns_tmp)
    if verbose:
      print(o, len(fns_tmp))

  for o in folders:
    fns_tmp = get_image_files(path, folders=o)
    random.shuffle(fns_tmp)
    fns.extend(fns_tmp[:min_type])
  return fns

fns = get_equal_images(path, cloud_types, verbose=True)
print(len(fns))
```

    Cirrus 141
    Cirrocumulus 103
    Cirrostratus 98
    Altocumulus 189
    Altostratus 104
    Nimbostratus 100
    Stratocumulus 97
    Stratus 109
    Cumulus 117
    Cumulonimbus 183
    970


## Loading and Training

Make a data loader and train a model. Here we use the get_equal_images() function that gets an equal number of random photos from each category. In the DataBlock I reserve 20% of the dataset for validation. I have some data in a separate folder for testing. The DataBlock also resizes all images to equal 128 x 128 pixel squares. This avoids problems with portrait and landscape photos.



```python

clouds = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_equal_images, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
```


```python
# optional
dls = clouds.dataloaders(path, folders=cloud_types)
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](/images/cloud_learner_files/cloud_learner_7_0.png)
    


Below I make use of some of the useful tools in FastAI that performs data augmentation techniques. Specifically, the RandomResizedCrop() takes zoomed areas of a photo to add additional variability into the data. The aug_transforms() perform skewing in addition to the random crop. These augments increase the size of the dataset.



```python
clouds = clouds.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = clouds.dataloaders(path)

```

Here, I train the model (vision_learner) starting with the pre-trained image model resnet34. The is evaluated using the error_rate which is the 1 - accuracy calculated from mean of the difference between the predictions and the target. Lower is better for error rate.

Next, I used the FastAI learning rate finder to identify an optimum learning rate for training the model.


```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
print(learn.loss_func)
lr_min,lr_steep = learn.lr_find(suggest_funcs=(minimum, steep))
```

    /usr/local/lib/python3.7/dist-packages/torchvision/models/_utils.py:136: UserWarning: Using 'weights' as positional parameter(s) is deprecated since 0.13 and will be removed in 0.15. Please use keyword parameter(s) instead.
      f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional "
    /usr/local/lib/python3.7/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet34_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet34_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)


    FlattenedLoss of CrossEntropyLoss()




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](/images/cloud_learner_files/cloud_learner_11_4.png)
    



```python
print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")
learn.fine_tune??
```

    Minimum/10: 1.00e-02, steepest point: 9.12e-03


Use the learning rate at the steepest point in curve and at a smaller learning rate than the minimum in loss. Here 9e-3.


```python
# optional
learn.fine_tune(3, base_lr=9e-3)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.612424</td>
      <td>2.310173</td>
      <td>0.495968</td>
      <td>03:01</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.496125</td>
      <td>1.906396</td>
      <td>0.431452</td>
      <td>01:26</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.340133</td>
      <td>1.347639</td>
      <td>0.310484</td>
      <td>01:28</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.076534</td>
      <td>1.151890</td>
      <td>0.294355</td>
      <td>01:29</td>
    </tr>
  </tbody>
</table>


Now try a different method to tune the top layer and then unfreeze to tune the bottom layers with a new learning rate.


```python
# optional
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 9e-3)
```

    /usr/local/lib/python3.7/dist-packages/torchvision/models/_utils.py:136: UserWarning: Using 'weights' as positional parameter(s) is deprecated since 0.13 and will be removed in 0.15. Please use keyword parameter(s) instead.
      f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional "
    /usr/local/lib/python3.7/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet34_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet34_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.495358</td>
      <td>2.492209</td>
      <td>0.500000</td>
      <td>01:24</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.027800</td>
      <td>1.577042</td>
      <td>0.431452</td>
      <td>01:22</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.641451</td>
      <td>1.216436</td>
      <td>0.379032</td>
      <td>01:23</td>
    </tr>
  </tbody>
</table>



```python
# optional
learn.unfreeze()
learn.lr_find()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    SuggestedLRs(valley=0.00015848931798245758)




    
![png](/images/cloud_learner_files/cloud_learner_17_3.png)
    


Use the best learning rate for the top layer and use a descriminative learning rate from 1e-5 to 1e-3 for the bottom layers.




```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(4, 9e-3)
learn.unfreeze()
learn.fit_one_cycle(10, lr_max=slice(1e-4, 5e-3))
```

    /usr/local/lib/python3.7/dist-packages/torchvision/models/_utils.py:136: UserWarning: Using 'weights' as positional parameter(s) is deprecated since 0.13 and will be removed in 0.15. Please use keyword parameter(s) instead.
      f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional "
    /usr/local/lib/python3.7/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet34_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet34_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.604423</td>
      <td>2.170120</td>
      <td>0.431452</td>
      <td>01:25</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.066355</td>
      <td>1.739304</td>
      <td>0.467742</td>
      <td>01:26</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.653228</td>
      <td>1.358257</td>
      <td>0.370968</td>
      <td>01:23</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.062416</td>
      <td>1.258690</td>
      <td>0.375000</td>
      <td>01:25</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.991020</td>
      <td>1.213077</td>
      <td>0.354839</td>
      <td>01:24</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.023106</td>
      <td>1.199777</td>
      <td>0.346774</td>
      <td>01:25</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.987026</td>
      <td>1.135217</td>
      <td>0.338710</td>
      <td>01:26</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.934739</td>
      <td>1.110483</td>
      <td>0.322581</td>
      <td>01:27</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.896102</td>
      <td>1.109210</td>
      <td>0.314516</td>
      <td>01:25</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.878263</td>
      <td>1.101606</td>
      <td>0.314516</td>
      <td>01:25</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.859027</td>
      <td>1.089587</td>
      <td>0.314516</td>
      <td>01:28</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.826027</td>
      <td>1.093557</td>
      <td>0.310484</td>
      <td>01:25</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.805568</td>
      <td>1.086081</td>
      <td>0.314516</td>
      <td>01:26</td>
    </tr>
  </tbody>
</table>


The plot below shows the number of training batches against the loss. While both the train and validation datasets have decreasing loss over the training cycles overfitting is not a problem. Towards the end of the training epochs the validation set loss flattens off suggesting additional training is not improving the model and is beginning to overfit to the training set.



```python
learn.recorder.plot_loss()
```


    
![png](/images/cloud_learner_files/cloud_learner_21_0.png)
    


The confusion matrix below shows the number of predictions by the model against the actual label of the photo for each of the categories for the validation dataset. Note, the uneven number of samples in each category is because of the random 20% in training and validation data.

The model does a fairly good job of predicting the cloud type for most of the categories. The category it struggles with are Altocumulus clouds that are confused for cirrocumulus. Both are upper upper atmosphere fluffy clouds. This suggests the model really needs more context to get the height of the cloud and that appearance alone makes it hard to distinguish to two.


```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](/images/cloud_learner_files/cloud_learner_23_4.png)
    


## Clean the Data

Now I have a trained model I can use the model to check the dataset for bad files that are not of cloud or wrongly labelled. To do this, FastAI has some useful functions for plotting the images that are confused and have a high loss rate. This means I can double check incorrect labels. Any that are bad images can be deleted or wrong labels can be re-labelled.

With cleaner data I then re-train the model in the previous step and iterate a bit.


```python
interp.most_confused(min_val=5)
interp.plot_top_losses(10, nrows=2)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](/images/cloud_learner_files/cloud_learner_25_2.png)
    



```python
cleaner = ImageClassifierCleaner(learn)
cleaner
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    VBox(children=(Dropdown(options=('Altocumulus', 'Altostratus', 'Cirrocumulus', 'Cirrostratus', 'Cirrus', 'Cumu…


Delete bad photos or change the label. Re-run trainer after this is needed.


```python
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
#for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```

## Export the model

After iterating on the pervious code to remove or relabel wrongly labelled data, I export the model.


```python
learn.export(fname='gdrive/MyDrive/Data_for_apps/Cloud_Learner/export.pkl')

!ls gdrive/MyDrive/Data_for_apps/Cloud_Learner
```

    clouds	export.pkl


## Conclusion

Using the script [here](https://github.com/benbardev/Cloud_Classifier/blob/main/src/cloud_classifier.ipynb), I took the trained model and deploy it for inference using the cloud application platform [Heroku](https://www.heroku.com/). Unfortunately, the free service they offered is no longer available but I may use Heroku again for a different project. This was interesting for me to take a project all the way to the deployment stage and have a little web application that could do something useful and test out my work.

This project had some challenges with finding useful data and relied on me to label some of it and check it. This was time consuming and may have some inaccuracies in my own cloud identification ability. The final dataset was on the small side. The result is a model that can classify photos of clouds.
