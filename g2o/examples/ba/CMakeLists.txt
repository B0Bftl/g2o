add_executable(ba_demo
  ba_demo.cpp
)

add_executable(ba_benchmark
        ba_benchmark.cpp)

add_executable(ba_demo_block
    ba_demo_block.cpp)

add_executable(test_solver
        test_solver.cpp)


include_directories(${CHOLMOD_INCLUDE_DIR})

target_link_libraries(ba_demo core solver_cholmod types_sba solver_eigen)

target_link_libraries(ba_demo_block core solver_cholmod types_sba )

target_link_libraries(ba_benchmark core solver_cholmod solver_eigen types_sba g2o_cli_library)

target_link_libraries(test_solver core solver_cholmod types_sba solver_eigen)
