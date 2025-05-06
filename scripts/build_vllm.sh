VLLM_HOME=$1
XFORMER_HOME=$2

cd $XFORMER_HOME
python setup.py install

cd $VLLM_HOME
pip install --no-build-isolation  -e .
