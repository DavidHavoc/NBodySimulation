#!/bin/bash

set -e

# configuration
SFML_VERSION=2.6.1
INSTALL_PREFIX=$HOME/SFML

echo "=== Downloading SFML $SFML_VERSION sources ==="
cd $HOME
wget -O SFML-${SFML_VERSION}-sources.zip https://www.sfml-dev.org/files/SFML-${SFML_VERSION}-sources.zip

echo "=== Extracting ==="
unzip -q SFML-${SFML_VERSION}-sources.zip
cd SFML-${SFML_VERSION}

echo "=== Creating build directory ==="
rm -rf build
mkdir build
cd build

echo "=== Running CMake with system GCC/G++ ==="
env LD_LIBRARY_PATH= cmake .. \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++

echo "=== Building SFML ==="
env LD_LIBRARY_PATH= make -j$(nproc)

echo "=== Installing SFML to ${INSTALL_PREFIX} ==="
env LD_LIBRARY_PATH= make install

echo "=== Cleaning up ==="
cd $HOME
rm -rf SFML-${SFML_VERSION} SFML-${SFML_VERSION}-sources.zip

echo "=== SFML installed in ${INSTALL_PREFIX} ==="
echo
echo "== Example usage =="
echo
echo "Compile your programs with:"
echo '  g++ myprogram.cpp -I$HOME/SFML/include -L$HOME/SFML/lib64 -Wl,-rpath,$HOME/SFML/lib64 -lsfml-graphics -lsfml-window -lsfml-system'
echo
echo "Or use this in your Makefile:"
cat <<EOF

CXX = /usr/bin/g++
CXXFLAGS = -std=c++17 -O2 -I\$(HOME)/SFML/include
LDFLAGS = -L\$(HOME)/SFML/lib64 -Wl,-rpath,\$(HOME)/SFML/lib64 -lsfml-graphics -lsfml-window -lsfml-system

TARGET = myprogram
SRC = myprogram.cpp

all: \$(TARGET)

\$(TARGET): \$(SRC)
	\$(CXX) \$(CXXFLAGS) \$^ -o \$@ \$(LDFLAGS)

run: \$(TARGET)
	LD_LIBRARY_PATH=\$(HOME)/SFML/lib64 ./\$(TARGET)

clean:
	rm -f \$(TARGET)
EOF

echo
echo "== To run your program =="
echo "  export LD_LIBRARY_PATH=\$HOME/SFML/lib64"
echo "  ./myprogram"
echo
echo "=== Done! ==="

