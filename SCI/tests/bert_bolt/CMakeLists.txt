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

add_executable(BOLT_BERT "bolt_bert.cpp")
target_link_libraries(BOLT_BERT LIB_BERT)


# add_executable(BOLT_WE_BERT "bolt_bert_word_elimination.cpp")
# target_link_libraries(BOLT_WE_BERT LIB_BERT)

add_executable(mirt_bolt mirt_bolt.cpp)
# Pull in the fixed-point implementation sources directly so we don't depend
# on SCI-FIX / SCI-UTILS library targets existing in this branch.
target_sources(mirt_bolt PRIVATE
  ${CMAKE_SOURCE_DIR}/src/library_fixed.cpp
  ${CMAKE_SOURCE_DIR}/src/cleartext_library_fixed.cpp
  ${CMAKE_SOURCE_DIR}/src/library_fixed_uniform.cpp
  ${CMAKE_SOURCE_DIR}/src/globals.cpp
  ${CMAKE_SOURCE_DIR}/src/globals_float.cpp
  ${CMAKE_SOURCE_DIR}/src/BuildingBlocks/truncation.cpp
  ${CMAKE_SOURCE_DIR}/src/BuildingBlocks/aux-protocols.cpp
  ${CMAKE_SOURCE_DIR}/src/BuildingBlocks/value-extension.cpp
)

# Link the libs that definitely exist (per your log).
target_link_libraries(mirt_bolt PRIVATE SCI-OT)

# You might also need these on some systems:
# target_link_libraries(mirt_bolt PRIVATE pthread gmp crypto)



