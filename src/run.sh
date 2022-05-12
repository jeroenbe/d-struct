echo "Starting exp 1"
python -m src.experiments --epochs=5 --n=200 --K=6 --nt-h_tol=1e-10 --nt-rho_max=1e+18
echo "Starting exp 2"
python -m src.experiments --epochs=5 --n=1000 --K=6 --nt-h_tol=1e-10 --nt-rho_max=1e+18

echo "Starting exp 3"
python -m src.experiments --epochs=5 --n=200 --K=3 --nt-h_tol=1e-10 --nt-rho_max=1e+18
echo "Starting exp 4"
python -m src.experiments --epochs=5 --n=1000 --K=3 --nt-h_tol=1e-10 --nt-rho_max=1e+18