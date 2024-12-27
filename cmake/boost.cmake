option(USE_SYSTEM_BOOST "Use system Boost" OFF)

if (DEFINED BOOST_ROOT)
  message(STATUS "Use Boost: ${BOOST_ROOT}")
elseif(USE_SYSTEM_BOOST)
  message(STATUS "Use system Boost")  
elseif(DEFINED CONDA_PREFIX)
  SET(BOOST_ROOT ${CONDA_PREFIX})
  SET(Boost_NO_SYSTEM_PATHS ON)
  message(STATUS "Use conda Boost: ${CONDA_PREFIX}")
endif()

find_package(Boost 1.80 REQUIRED COMPONENTS python)

message(STATUS "Find Boost ${Boost_VERSION}: ${Boost_INCLUDE_DIR}")