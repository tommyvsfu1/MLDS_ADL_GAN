
def add_argument(parser):
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'Anime'],help='The name of dataset')
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights, WGAN paper default values")
    parser.add_argument("--sample_interval", type=int, default=5000, help="interval betwen image samples")
    parser.add_argument("--model_use",type=str,default='WGAN',help="WGAN model")
    return parser
