if (DEFINED CONDA_PREFIX)
  SET(BOOST_ROOT ${CONDA_PREFIX})
  SET(Boost_NO_SYSTEM_PATHS ON)
endif()

find_package(Boost 1.80 REQUIRED)

message(STATUS "Find Boost ${Boost_VERSION}: ${Boost_INCLUDE_DIR}")