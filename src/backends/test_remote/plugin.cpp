// SPDX-License-Identifier: MIT
// ii-backend-test-remote: the simulated device of test_remote.h as a backend plugin.
//
// Options: "test_devices" (the device specification ii got through ContextOptions::test_devices or II_TEST_DEVICES;
// empty: no device) and "test_remote.unavailable" ("1": report the plugin as unavailable, for tests).
#include <ii/version.h>

#include "backends/test_remote/test_remote.h"
#include "plugin/sdk.h"

namespace {

using ii::plugin::BackendList;

ii::Result<BackendList> create_backends(const ii::plugin::PluginHost& host) {
    if (host.option("test_remote.unavailable") == "1") {
        return ii::errors::unavailable("unavailable on request (test_remote.unavailable=1)");
    }
    const std::string spec = host.option("test_devices");
    BackendList backends;
    if (spec.empty() || spec == "none") {
        return backends;
    }
    II_ASSIGN_OR_RETURN(std::vector<ii::test_remote::TestDeviceOptions> devices,
                        ii::test_remote::parse_test_devices(spec));
    for (const ii::test_remote::TestDeviceOptions& device : devices) {
        backends.push_back(ii::test_remote::create_test_remote_backend(device));
    }
    return backends;
}

constexpr ii::plugin::PluginDefinition kPlugin{"ii-backend-test-remote", II_VERSION_STRING, &create_backends};

}  // namespace

II_DEFINE_BACKEND_PLUGIN(kPlugin)
