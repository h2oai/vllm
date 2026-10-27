# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H2O: the model package's decision API, served by this server.

``vllm serve h2oai/h2o-lightning-4b --h2o-decisions`` exposes ``/v1/decisions``,
``/v1/systemone``, ``/api/alpha/decisions`` and ``/health`` exactly as the
package's released shim does. The shim module shipped in the model repository
runs inside this process: its own ``main()`` builds it from the package's
``serve_config*.json`` (bridge-driven on in-memory requests, see
``bridge.py``), and its calls to "vLLM" go to this server's serving handlers
(``transport.py``) instead of a loopback socket. Nothing of the prompt render,
the readout, the temperatures, the probes, Box, the images or the state cut is
re-derived here. See ``tools/h2o/DECISIONS_ENDPOINT.md``.
"""
