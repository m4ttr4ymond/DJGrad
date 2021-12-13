# python test_attack.py --gpu 0 --attack invert_loss --mode none
# python test_attack.py --gpu 0 --attack invert_loss --mode add
# python test_attack.py --gpu 0 --attack invert_loss --mode djgrad
python test_attack.py --gpu 7 --attack backdoor --mode none
python test_attack.py --gpu 7 --attack backdoor --mode add
python test_attack.py --gpu 7 --attack backdoor --mode djgrad
