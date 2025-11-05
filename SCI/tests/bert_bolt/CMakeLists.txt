add_library(LIB_BERT 
    he.cpp
    nonlinear.cpp
    linear.cpp
    bert_utils.cpp
    bert.cpp
    )

target_link_libraries(LIB_BERT SCI-FloatingPoint)
target_link_libraries(LIB_BERT SCI-HE)

# add_executable(BERT_NONLINEAR "test_nonlinear.cpp")
# target_link_libraries(BERT_NONLINEAR NONLINEAR)

add_executable(mirt_bolt tests/bert_bolt/mirt_bolt.cpp)

target_sources(mirt_bolt PRIVATE
  ${CMAKE_SOURCE_DIR}/src/library_fixed.cpp
  ${CMAKE_SOURCE_DIR}/src/cleartext_library_fixed.cpp
  ${CMAKE_SOURCE_DIR}/src/library_fixed_uniform.cpp
  ${CMAKE_SOURCE_DIR}/src/globals.cpp          # keep
  # DO NOT add globals_float.cpp (it redefines iopackArr/otpackArr)
  ${CMAKE_SOURCE_DIR}/src/BuildingBlocks/truncation.cpp
  ${CMAKE_SOURCE_DIR}/src/BuildingBlocks/aux-protocols.cpp
  ${CMAKE_SOURCE_DIR}/src/BuildingBlocks/value-extension.cpp
)

target_link_libraries(mirt_bolt PRIVATE SCI-OT)  # this exists in your build
# If needed on your system:
# target_link_libraries(mirt_bolt PRIVATE pthread gmp crypto)




