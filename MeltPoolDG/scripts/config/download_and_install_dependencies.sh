#number of processes
np=${1:-4}
buildConfig=${2:"DebugRelease"}

#config dir (MeltPoolDG dependening on the location of this file)
configDir=$(dirname -- "$0")
configDir=$(realpath "$configDir")

# Get the absolute path of the script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/log.sh

# TODO: extract checks

##############################################################
# check proper cmake version
##############################################################
cmp=3.17.0
ver=$(cmake --version | head -1 | cut -f3 -d" ")

mapfile -t sorted < <(printf "%s\n" "$ver" "$cmp" | sort -V)

if [[ ${sorted[0]} == "$cmp" ]]; then
    log "cmake version $ver >= $cmp"
else
    log "ERROR: cmake version too low; update to at least $cmp."
    exit 1
fi

##############################################################
# check gcc version
##############################################################
# Get the GCC version
gcc_version=$(gcc --version | head -n 1 | awk '{print $3}')

# Extract the major version
gcc_major_version=$(log $gcc_version | cut -d'.' -f1)

##############################################################
# check if metis is installed
##############################################################
check_metis() {
  ldconfig -p | grep libmetis

  if [[ $(ldconfig -p | grep libmetis) ]]; then
      log "Dependency libmetis found."
  else
      log "WARNING: Dependency libmetis not found. Make sure to install metis if you want to use the functionalities."
      exit 1
  fi
}

check_metis # Call function

##############################################################
# check if boost is installed
##############################################################
# Function to check if Boost is installed
check_boost() {
    if ldconfig -p | grep -q libboost; then
        log "Boost is installed."
    elif [[ -f "/usr/include/boost/version.hpp" || -f "/usr/local/include/boost/version.hpp" ]]; then
        log "Boost headers found."
    elif pkg-config --exists boost; then
        log "Boost detected via pkg-config."
    else
        log "ERROR: Boost library is missing. Install Boost before proceeding."
        exit 1
    fi
}

check_boost # Call function

##############################################################
# check if blas is installed
##############################################################
check_blas() {
    if ldconfig -p | grep -E 'libblas\.so|libopenblas\.so' > /dev/null; then
        log "BLAS is installed."
    else
        log "ERROR: BLAS is not installed. Please install BLAS before proceeding."
        exit 1
    fi
}

check_blas  # Call function

##############################################################
# install p4est
##############################################################
wget http://p4est.github.io/release/p4est-2.2.tar.gz
mkdir -p `$(pwd)/p4est-build`
$configDir/p4est-config.sh p4est-2.2.tar.gz `$(pwd)/p4est-build`
rm p4est-2.2.tar.gz

###############################################################
## install Trilinos
###############################################################
mkdir -p trilinos-build

if [ "$gcc_major_version" -lt 13 ]; then
  log "Use patched Trilinos version 13.4.1."
  wget https://github.com/trilinos/Trilinos/archive/trilinos-release-13-4-1.tar.gz
  mv trilinos-release-13-4-1.tar.gz trilinos-release-13-4-1.tar
  tar -xvf trilinos-release-13-4-1.tar
  cd trilinos-build
  rm -rf *
  $configDir/trilinos-config.sh ../Trilinos-trilinos-release-13-4-1
  rm ../trilinos-release-13-4-1.tar
else
  log "Use patched Trilinos version 13.4.1 for GCC13"
  wget https://github.com/MeltPoolDG/Trilinos/archive/trilinos-release-13-4-1-for-gcc-13.zip
  unzip trilinos-release-13-4-1-for-gcc-13.zip
  cd trilinos-build
  rm -rf *
  $configDir/trilinos-config.sh ../Trilinos-trilinos-release-13-4-1-for-gcc-13
  rm ../trilinos-release-13-4-1-for-gcc-13.zip
fi

make -j$np install
cd ..

###############################################################
## install deal.II
###############################################################
git clone https://github.com/dealii/dealii
cd dealii
git checkout b369f187d84c3d322ca2f9cc20239151edd9ba64
cd ..
mkdir -p dealii-build
cd dealii-build
rm -rf *
$configDir/dealii-config.sh ../trilinos-install ../p4est-install ../dealii $buildConfig
make -j$np
cd ..

##############################################################
# install adaflo
##############################################################
git clone https://github.com/MeltPoolDG/adaflo
# release
cd adaflo
git checkout eed1849922bb32c4d06a769a8a28b80eb863421c
if [[ "$buildConfig" == "Release" || "$buildConfig" == "DebugRelease" ]]; then
  mkdir -p build_release
  cd build_release
  cmake -DBUILD_SHARED_LIBS=true -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=../dealii-build -DCMAKE_CXX_COMPILER=mpicxx ..
  make -j$np adaflo
  cd ..
fi
# debug
if [[ "$buildConfig" == "Debug" || "$buildConfig" == "DebugRelease" ]]; then
  mkdir -p build_debug
  cd build_debug
  cmake -DBUILD_SHARED_LIBS=true -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=../dealii-build -DCMAKE_CXX_COMPILER=mpicxx ..
  make -j$np adaflo
  cd ..
fi

log ""
log "Dependencies successfully installed. You may add the folders to your path via"
log "log 'export PATH=$(realpath $pwd/dealii-build):$PATH' >> ~/.bashrc"
log "log 'export ADAFLO_INCLUDE=$(realpath $pwd/adaflo/include):$PATH' >> ~/.bashrc"
