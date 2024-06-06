message(STATUS "ONERT MICRO DEV BUILD BEGIN")

set(SOURCES
    onert-micro.cpp)

add_library(${OM_DEV_LIB} STATIC ${SOURCES})

target_include_directories(${OM_DEV_LIB} PUBLIC "${OM_INCLUDE_DIR}")
target_link_libraries(${OM_DEV_LIB} PUBLIC ${OM_INTERPRETER_LIB})

message(STATUS "ONERT MICRO DEV BUILD FINISHED")
