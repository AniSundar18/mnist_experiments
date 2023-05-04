# training scripts
#CUDA_VISIBLE_DEVICES=3 python3 new_main.py --model CNN --train_datasets mnist mnist_cbg mnistm --test_datasets mnist mnist_cbg mnistm --save-model --epochs 15 --SEED 2
#CUDA_VISIBLE_DEVICES=3 python3 new_main.py --model CNN --teacher_model CNN --train_datasets mnist --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm_CNN  --distill kd --save-model  --epochs 15 --SEED 2

#CUDA_VISIBLE_DEVICES=3 python3 new_main.py --model CNN --train_datasets mnist --test_datasets mnist mnist_cbg mnistm --save-model  --epochs 15 --SEED 2

#CUDA_VISIBLE_DEVICES=3 python3 new_main.py --model ViT --train_datasets mnist mnist_cbg mnistm --test_datasets mnist mnist_cbg mnistm --save-model --epochs 15 --SEED 2
CUDA_VISIBLE_DEVICES=3 python3 new_main.py --model CNN --teacher_model CNN --train_datasets mnistm --test_datasets mnist mnist_cbg mnistm --save-model  --epochs 15 --pretrained_model mnist_mnist_cbg_mnistm_CNN --distill hint --SEED 2
#CUDA_VISIBLE_DEVICES=3 python3 new_main.py --model CNN --teacher_model ViT --train_datasets mnistm --test_datasets mnist mnist_cbg mnistm --save-model  --epochs 15 --SEED 2
#CUDA_VISIBLE_DEVICES=3 python3 new_main.py --model CNN --train_datasets mnistm --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm_CNN  --distill hint --save-model  --epochs 15 --SEED 2

#CUDA_VISIBLE_DEVICES=3 python3 main.py --train_datasets mnistm --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm  --save-model --distill kd --epochs 15 --SEED 2

#CUDA_VISIBLE_DEVICES=3 python3 main.py --train_datasets mnistm --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm  --save-model --epochs 15 --SEED 2

#CUDA_VISIBLE_DEVICES=3 python3 main.py --train_datasets mnist --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm
#--save-model --epochs 50

#CUDA_VISIBLE_DEVICES=3 python3 main.py --train_datasets mnist_cbg --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm  --save-model --epochs 50

#CUDA_VISIBLE_DEVICES=3 python3 main.py --train_datasets mnistm --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm  --save-model --epochs 50
#CUDA_VISIBLE_DEVICES=3 python3 main.py --train_datasets mnist --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm --distill kd --save-model --epochs 50

#CUDA_VISIBLE_DEVICES=3 python3 main.py --train_datasets mnist_cbg --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm  --distill kd --save-model --epochs 50

#CUDA_VISIBLE_DEVICES=3 python3 main.py --train_datasets mnistm --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm  --distill kd --save-model --epochs 50


#CUDA_VISIBLE_DEVICES=3 python3 main.py --train_datasets mnist mnist_cbg mnist_cfg mnistm --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnistm  --save-model --epochs 50

#CUDA_VISIBLE_DEVICES=3 python3 main.py --train_datasets mnist_cfg --test_datasets mnist mnist_cbg mnistm --pretrained_model mnist_mnist_cbg_mnist_cfg_mnistm_student --train-few --distill kd --save-model --epochs 50
