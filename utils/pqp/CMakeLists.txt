add_library(pqp STATIC
    Build.cpp
    BV.cpp
    Distance_OBB.cpp
    PQP.cpp
    TriDist.cpp
)

target_include_directories(pqp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(pqp PUBLIC Eigen3::Eigen)
