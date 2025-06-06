if (NOT BUILD_TRIX_LOADER)
  return()
endif ()

find_package(PkgConfig REQUIRED)
# TRIXEngine version is required to higher than 2.5.0
pkg_check_modules(TRIXEngine QUIET IMPORTED_TARGET npu-engine>2.5.0)

if(TRIXEngine_FOUND)
  message(STATUS "ONERT frontend: Found TRIXEngine")
  list(APPEND SOURCES TrixLoader.cc)
else()
  message(STATUS "ONERT frontend: Failed to find TRIXEngine")
  return()
endif(TRIXEngine_FOUND)

# Loader library name should be lib<type>_loader.so
add_library(tvn_loader SHARED ${SOURCES})
set_target_properties(tvn_loader PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
target_link_libraries(tvn_loader PRIVATE onert_core)
target_link_libraries(tvn_loader PRIVATE nnfw_common nnfw_coverage)
target_link_libraries(tvn_loader PRIVATE PkgConfig::TRIXEngine)

install(TARGETS tvn_loader DESTINATION lib/nnfw/loader)
