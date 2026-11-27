# SPDX-License-Identifier: MIT

# A simulated device for testing multi-device plans (see test_remote.h); registered only on request.
ii_add_component(backend_test_remote SOURCES
    test_remote.cpp)

# The same backend as a plugin (ii-backend-test-remote): tests the plugin interface. When the plugin is loaded
# (II_PLUGIN_PATH=<build>/plugins/<config>), it provides the simulated devices instead of the built-in backend.
if(II_BUILD_TESTS)
    ii_add_backend_plugin(ii-backend-test-remote SOURCES plugin.cpp)
endif()
