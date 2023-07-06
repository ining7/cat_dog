@echo off
call conda activate deploy

echo Running with lr=0.01, batch_size=100, epochs=40
set lr=0.01
set batch_size=100
set epochs=40
set model_name=../check_point/train_10_100_40.pth
set save_path=../res/train_10_100_40.png
python training.py %lr% %batch_size% %epochs% %model_name% %save_path% > res\output_10_100_40.txt

echo Running with lr=0.01, batch_size=200, epochs=20
set lr=0.01
set batch_size=200
set epochs=20
set model_name=../check_point/train_10_200_20.pth
set save_path=../res/train_10_200_20.png
python training.py %lr% %batch_size% %epochs% %model_name% %save_path% > res\output_10_200_20.txt

echo Running with lr=0.01, batch_size=300, epochs=20
set lr=0.01
set batch_size=300
set epochs=20
set model_name=../check_point/train_10_300_20.pth
set save_path=../res/train_10_300_20.png
python training.py %lr% %batch_size% %epochs% %model_name% %save_path% > res\output_10_300_20.txt

echo Running with lr=0.005, batch_size=300, epochs=20
set lr=0.005
set batch_size=300
set epochs=20
set model_name=../check_point/train_5_300_20.pth
set save_path=../res/train_5_300_20.png
python training.py %lr% %batch_size% %epochs% %model_name% %save_path% > res\output_5_300_20.txt

echo Running with lr=0.005, batch_size=500, epochs=50
set lr=0.005
set batch_size=500
set epochs=50
set model_name=../check_point/train_5_500_50.pth
set save_path=../res/train_5_500_50.png
python training.py %lr% %batch_size% %epochs% %model_name% %save_path% > res\output_5_500_50.txt

echo All tasks completed!
