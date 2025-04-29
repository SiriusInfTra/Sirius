option(USE_SYSTEM_BOOST "Use system Boost" OFF)

if (DEFINED Boost_ROOT)
  message(STATUS "Use Boost: ${Boost_ROOT}")
  find_package(Boost 1.80 REQUIRED 
    HINTS ${Boost_ROOT}
    PATHS ${Boost_ROOT}
    COMPONENTS python json
  )
elseif(USE_SYSTEM_BOOST)
  message(STATUS "Use system Boost")
  find_package(Boost 1.80 REQUIRED COMPONENTS python json)
elseif(DEFINED CONDA_PREFIX)
  SET(Boost_ROOT ${CONDA_PREFIX})
  SET(Boost_NO_SYSTEM_PATHS ON)
  message(STATUS "Use conda Boost: ${CONDA_PREFIX}")
  find_package(Boost 1.80 REQUIRED COMPONENTS python json)
endif()

message(STATUS "${CMAKE_PROJECT_NAME}: Find Boost ${Boost_VERSION}: ${Boost_DIR}")