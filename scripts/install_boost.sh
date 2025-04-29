boost_root_dir="$1"

url="https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz"
boost_tarball_name=$(basename "$url")
boost_dir_name="boost_1_82_0"

# Check if boost_root_dir parameter is provided
if [ -z "$boost_root_dir" ]; then
    echo "Error: Please provide a directory for Boost installation"
    echo "Usage: $0 <boost_root_dir>"
    exit 1
fi

# Create the root directory if it doesn't exist
mkdir -p "$boost_root_dir"
echo "Using installation directory: $boost_root_dir"

# Change to the root directory
cd "$boost_root_dir"

# Download Boost
echo "Downloading Boost from $url"
if command -v wget &> /dev/null; then
    wget "$url" -O "$boost_tarball_name"
elif command -v curl &> /dev/null; then
    curl -L "$url" -o "$boost_tarball_name"
else
    echo "Error: Neither wget nor curl is available. Please install one of them."
    exit 1
fi

# Extract the tarball
echo "Extracting Boost"
tar -xzf "$boost_tarball_name"

# Change to the Boost directory
cd "$boost_dir_name"

# Configure Boost
echo "Configuring Boost"
./bootstrap.sh --prefix="$boost_root_dir/install"

# Determine number of CPU cores for parallel compilation
if command -v nproc &> /dev/null; then
    NUM_CORES=$(nproc)
else
    NUM_CORES=4  # Default to 4 if we can't determine
fi

# Build Boost
echo "Building Boost with $NUM_CORES parallel jobs (this may take a while)"
./b2 -j "$NUM_CORES"

# Install Boost
echo "Installing Boost to $boost_root_dir/install"
./b2 install --prefix="$boost_root_dir/install"

echo "Boost has been successfully compiled and installed in $boost_root_dir/install"
echo "Set Boost_ROOT=$boost_root_dir/install in your environment to use this Boost installation"
