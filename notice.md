# Need to check after

## Warning

The file is corrupted, please download it again from the following link:
`/workspace/navsim_workspace/dataset/navsim_logs/trainval/2021.06.09.17.23.18_veh-38_00305_00597.pkl`

## Need installed

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

Also need to installed diffusers



There are 2 sensorblobs causing issues
```
dio@ae0ec7ce6597:/workspace/DiffusionDrive$ ls navsim_workspace/dataset/navsim_logs/trainval/ | wc -l
1310
dio@ae0ec7ce6597:/workspace/DiffusionDrive$ ls navsim_workspace/dataset/sensor_blobs/ | wc -l
sensor_blobs/ trainval/     
dio@ae0ec7ce6597:/workspace/DiffusionDrive$ ls navsim_workspace/dataset/sensor_blobs/sensor_blobs/ | wc -l
1
dio@ae0ec7ce6597:/workspace/DiffusionDrive$ ls navsim_workspace/dataset/sensor_blobs/sensor_blobs/trainval/ | wc -l
1310
dio@ae0ec7ce6597:/workspace/DiffusionDrive$ cd navsim
navsim/           navsim.egg-info/  navsim_workspace/ 
dio@ae0ec7ce6597:/workspace/DiffusionDrive$ cd navsim_workspace/
dataset/ exp/     navsim/  
dio@ae0ec7ce6597:/workspace/DiffusionDrive$ ls navsim_workspace/dataset/sensor_blobs/trainval/ | wc -l
20
```
in evaluation need to escape the model 