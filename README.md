If you encounter any difficulties while running the code, please don’t hesitate to reach out to me. I will do my best to provide full support.

## Environments

conda create -n ver python=3.7.16  
conda activate ver  
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia  
Install other packages: pip install -r requirements.txt  



## Datasets
Get beer-related datasets [here](http://people.csail.mit.edu/taolei/beer/). Then place it in the ./data/beer directory.  
Get hotel-related datasets [here](https://people.csail.mit.edu/yujia/files/r2a/data.zip). 
Then  find hotel_Location.train, hotel_Location.dev, hotel_Service.train, hotel_Service.dev, hotel_Cleanliness.train, hotel_Cleanliness.dev from data/oracle and put them in the ./data/hotel directory. 
Find hotel_Location.train, hotel_Service.train, hotel_Cleanliness.train from data/target and put them in the ./data/hotel/annotations directory.  
Word embedding: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/). Then put it in the ./data/hotel/embeddings directory.


## Running example
You can use:      
python -u perturbation.py --seed $seed --perturb_rate 0.3 --dis_lr 0 --lr 0.0001 --batch_size 128 --sparsity_percentage 0.125 --sparsity_lambda 11 --continuity_lambda 12 --epochs 300 --aspect 0
to get the results of Beer-Appearance. (Please replace $seed with a choice of [1,2,3,4,5])  


