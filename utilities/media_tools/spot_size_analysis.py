"""Camera spot size -> short-arm NA, through the cavity-design simulation.

The mode that leaves the cavity through the Coastline mirror and lands on the camera is the same
mode that goes the other way into the short arm, so its size on the camera fixes the NA there. The
propagation is simulated by the cavity-design project
(simple_analysis_scripts/camera_spot_size_per_cavity_NA.py); this module is the seam between it and
the measuring scripts (utilities/media_tools/postprocessing_camera_video.py): it forwards the
geometry, caches the scan, and turns "the cavity-design project is not available" into an error
message instead of an exception - a missing simulation must not cost the measured spot size.

Sibling of pico_scope/mode_analysis.py's NA section, which does the same for mode spacing.
"""

_interpolator_cache = {}  # geometry -> (spot_size_to_NA, error)


def _import_simulation():
    """Import the cavity-design camera-spot-size simulation (raises if unavailable)."""
    from utilities.cavity_design_bridge import import_cavity_design_module
    return import_cavity_design_module('simple_analysis_scripts.camera_spot_size_per_cavity_NA')


def list_cavity_elements():
    """Names accepted in the element lists, from the cavity-design catalog.

        python -c "from utilities.media_tools.spot_size_analysis import list_cavity_elements; print(*list_cavity_elements(), sep='\\n')"
    """
    return _import_simulation().available_element_names()


def get_spot_size_to_na(long_arm_length, mid_arm_length=None, lens_distance=None,
                        camera_distance=None, outgoing_elements=None, intracavity_elements=None,
                        N_points=None, NA_long_arm_range=None,
                        measured_spot_sizes_m=(), measured_labels=(), plot=False):
    """Return (spot_size_to_NA, error).

    spot_size_to_NA maps a camera spot size [m] (the 1/e^2 intensity radius w, which is what
    fit_gaussian's w_x / w_y are) to the NA in the cavity's short arm. Every distance is in metres;
    the ones left None keep the simulation's own default. `plot=True` shows the simulated system in
    one window - the two dependency panels plus the optical system underneath - with the spot sizes
    in `measured_spot_sizes_m` (named by `measured_labels`, e.g. ('w_x', 'w_y')) marked on it.

    `outgoing_elements` names what the beam meets between the cavity and the camera, starting at the
    end mirror in transmission; `intracavity_elements` names the lens (or two, `mid_arm_length`
    apart) it meets going back into the short arm. Both are catalog names in optical order - see
    list_cavity_elements(). A name that is not in the catalog is an error the caller must fix, so it
    is raised, not reported.

    `NA_long_arm_range` is the (min, max) long-arm NA the scan sweeps, and with it the range of
    camera spot sizes the result is defined over - widen it when a measured spot size comes back as
    out of range (it cannot start below the simulation's NA floor, where the mode stops matching the
    Coastline mirror).

    When the cavity-design project cannot be imported or the simulation fails, returns
    (None, '<why>') and the caller reports the spot sizes with the NA marked unavailable.
    """
    key = (long_arm_length, mid_arm_length, lens_distance, camera_distance, N_points,
           NA_long_arm_range,
           tuple(outgoing_elements) if outgoing_elements is not None else None,
           tuple(intracavity_elements) if intracavity_elements is not None else None)
    # The cache holds interpolators, not figures, and the measured spot sizes only change what is
    # drawn - so a caller that asked for a plot re-runs the scan, or a second run in the same
    # session (a live console re-running a script) would get no plot at all.
    if key in _interpolator_cache and not plot:
        return _interpolator_cache[key]
    try:
        simulation = _import_simulation()
    except Exception as error:
        result = (None, f'{type(error).__name__}: {error}')
    else:
        geometry = {'long_arm_length': long_arm_length, 'mid_arm_length': mid_arm_length,
                    'lens_distance': lens_distance, 'camera_distance': camera_distance,
                    'outgoing_elements': outgoing_elements,
                    'intracavity_elements': intracavity_elements,
                    'N_points': N_points, 'NA_long_arm_range': NA_long_arm_range}
        geometry = {name: value for name, value in geometry.items() if value is not None}
        try:
            spot_size_to_NA = simulation.make_spot_size_to_NA_interpolator(
                measured_spot_sizes_m=measured_spot_sizes_m, measured_labels=measured_labels,
                plot=plot, **geometry)
            result = (spot_size_to_NA, None)
        except simulation.UnknownCavityElement:
            # A misspelled element name is something the caller has to fix in its own config, not a
            # missing simulation. Fail loudly instead of silently dropping the NA from the report.
            raise
        except Exception as error:
            result = (None, f'{type(error).__name__}: {error}')
    _interpolator_cache[key] = result
    return result


def na_from_spot_size(spot_size_to_NA, camera_spot_size_m):
    """Return (NA, error) for one camera spot size [m].

    A spot size the simulated scan never produced gives (None, '<why>'): the interpolator refuses to
    extrapolate (SpotSizeOutOfRange, a ValueError), and a spot size the simulation cannot explain is
    a fact about the measurement worth printing, not a crash worth losing the fit over.
    """
    try:
        return float(spot_size_to_NA(camera_spot_size_m)), None
    except ValueError as error:
        return None, f'{type(error).__name__}: {error}'
