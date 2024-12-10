# dlcv_proj1
### Installation
```bash
conda create -n taco python=3.10.6
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

### Dataset
* Download the [TACO](http://tacodataset.org/) dataset or `python download_dataset.py`
* Change the paths to the script `util/mypath.py`

### Training
* `python extract_object_proposals.py --imset train`, to extract object proposals for each training image
* `python match_proposals.py --imset train`, to match each proposal to a ground-truth box or to background
* `python train.py`
