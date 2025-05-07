TVM_HOME=$1

cd $TVM_HOME/build
cmake .. -GNinja 
cmake --build .

cd $TVM_HOME/python
python setup.py install



