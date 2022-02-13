# CP2A Dataset: Carla Pedestrian Anticipation Dataset


<h1>This repository is dedicated for the CP2A dataset described in the following paper:
<b>'Is attention to bounding boxes all you need for pedestrian action prediction?'</b>
<p>
https://arxiv.org/abs/2107.08031
</p></h1>




<h1>Description </h1>
In this paper, we present a new simulated dataset for pedestrian action anticipation collected using the CARLA simulator.

To generate this dataset, we place a camera sensor on the ego-vehicle in the Carla environment and set the parameters to those of the camera used to record the PIE dataset (i.e., 1920x1080, 110° FOV). Then, we compute bounding boxes for each pedestrian interacting with the ego vehicle as seen through the camera's field of view.
We generated the data in two urban environments available in the CARLA simulator: Town02 and Town03.

The total number of simulated pedestrians is nearly 55k, equivalent to 14M bounding boxes samples.
The critical point for each pedestrian is their first point of crossing the street (in case they will eventually cross) or the last bounding box coordinates of their path in the opposite case.
The crossing behavior represents 25\% of the total pedestrians. We balanced the training split of the dataset to obtain labeled sequences crossing/non-crossing in equal parts. We used sequence-flipping to augment the minority class (i.e., crossing behavior in our case) and then undersampled the rest of the dataset. The result is a total of nearly 50k pedestrian sequences.

Next, the pedestrian trajectory sequences were transformed into observation sequences of equal length (i.e., 0.5 seconds) with a 60% overlap for the training splits. The TTE length is between 30 and 60 frames. It resulted in a total of nearly 220k observation sequences. 


<h1>Dataset Download and Preprocessing</h1>

1- Download the dataset from the following link:
https://drive.google.com/drive/folders/19l_20y83n_qACOH1GfX79jEqClKB6Pgl?usp=sharing

2- Place the dataset inside the data directory

3- Use the data/prepocessing.py file to process the dataset. As a result, we will get a training, validation, and testing datasets each composed of the bounding boxes coordinates of the pedestrians alongside their corresonding labels.

4- For using the dataset during training, use the data/dataloader.py file.


<h1>
<b>Authors</b>
</h1>

* [Lina Achaji](https://scholar.google.com/citations?user=RMO2zJAAAAAJ&hl=en)

Please send an email to lina.achaji@stellantis.com or lina.achaji@inria.fr if there are any problems with using the data or the code.



