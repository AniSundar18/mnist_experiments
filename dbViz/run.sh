python3 get_iou.py --net CNN --baseset MNIST --plane_datasets mnist --load_net dist_models/mnist_mnist_cbg_mnistm_ViT.pt --load_ind_net dist_models/mnistm_CNN_from_ViT.pt --load_dist_net dist_models/mnistm_CNN_from_ViT_kd.pt --teacher_net ViT_lucid

python3 get_iou.py --net CNN --baseset MNIST --plane_datasets mnist_cbg --load_net dist_models/mnist_mnist_cbg_mnistm_ViT.pt --load_ind_net dist_models/mnistm_CNN_from_ViT.pt --load_dist_net dist_models/mnistm_CNN_from_ViT_kd.pt --teacher_net ViT_lucid 

python3 get_iou.py --net CNN --baseset MNIST --plane_datasets mnistm --load_net dist_models/mnist_mnist_cbg_mnistm_ViT.pt --load_ind_net dist_models/mnistm_CNN_from_ViT.pt --load_dist_net dist_models/mnistm_CNN_from_ViT_kd.pt --teacher_net ViT_lucid 

#python3 main.py --net CNN --baseset MNIST --plane_datasets mnistm --load_net dist_models/mnist_mnist_cbg_mnistm_CNN.pt --load_ind_net dist_models/mnistm_CNN.pt --load_dist_net dist_models/mnistm_CNN_student.pt
