cc_client_build_dir=$1

cache_cmake=$cc_client_build_dir/tmp/cc-clients-cache-Release.cmake

if [ ! -f $cache_cmake ]; then
  echo "cannot find $cache_cmake"
  exit 1
fi

if [ $(grep -c "TRITON_KEEP_TYPEINFO" $cache_cmake) -ne 0 ]; then
  echo "TRITON_KEEP_TYPEINFO is already set"
  echo $(cat $cache_cmake | grep "TRITON_KEEP_TYPEINFO")
else
  echo "Setting TRITON_KEEP_TYPEINFO"
  echo -e "\nset(TRITON_KEEP_TYPEINFO \"ON\" CACHE BOOL \"Initial cache\" FORCE)" >> $cache_cmake
fi