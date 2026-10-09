import pytest

from frontfinder.config.manifests import (
    ALL_CLASSES,
    SERVED_CLASSES,
    SPIX_WCE_DILATION_MANIFEST,
    THETA_E_UV_Q_MANIFEST,
    MODEL_1702_MANIFEST,
    NBS_SQUARE_DILATION_TROF_TT_MANIFEST,
    ModelManifest,
    VariableSpec,
    get_manifest,
)


def test_theta_e_uv_q_channel_count_matches_training_config():
    # sooner_ablations.yaml originally listed n_channels: 30 (5 vars x 6
    # levels). potential_vorticity was dropped 2026-08-20 -- no native IFS
    # open-data PV field to fetch, and the isobaric-PV approximation had no
    # verified relationship to training's real ERA5-native PV.
    # equivalent_potential_temperature stays -- Taylor confirmed the
    # retrained _best_loss.keras still uses it. Now 4 vars x 6 levels = 24.
    assert THETA_E_UV_Q_MANIFEST.n_channels == 24


def test_model_1702_channel_count():
    # 5 pressure-level vars x 4 levels (20) + 5 single-level vars (5) = 25
    assert MODEL_1702_MANIFEST.n_channels == 25


def test_theta_e_uv_q_channel_names_are_level_major_variable_minor_in_order():
    # Matches fronts/src/fronts/data/inputs.py's inputs_ds_to_dataarray():
    # pressure-level channels stack LEVEL-outer, VARIABLE-inner (every
    # variable at level[0], then every variable at level[1], ...) -- NOT
    # each variable's own levels grouped together. Getting this backwards
    # was a real bug found via live smoke-testing on 2026-08-20 (see
    # ecmwf_ifs.py's assemble_model_input docstring): it fed the model's
    # baked-in per-channel normalization stats to the wrong channels and
    # silently produced garbage (background-saturated) predictions.
    names = THETA_E_UV_Q_MANIFEST.channel_names()
    assert names[0:4] == [
        "equivalent_potential_temperature_1000",
        "u_component_of_wind_1000",
        "v_component_of_wind_1000",
        "specific_humidity_1000",
    ]
    assert names[4:8] == [
        "equivalent_potential_temperature_925",
        "u_component_of_wind_925",
        "v_component_of_wind_925",
        "specific_humidity_925",
    ]
    assert names[20:24] == [
        "equivalent_potential_temperature_300",
        "u_component_of_wind_300",
        "v_component_of_wind_300",
        "specific_humidity_300",
    ]
    assert len(names) == 24
    assert len(set(names)) == 24  # no duplicate channels


def test_manifest_rejects_pressure_variables_with_mismatched_levels():
    # level-major channel ordering is only well-defined when every
    # pressure-level variable shares the same level list -- see
    # ModelManifest.__post_init__.
    with pytest.raises(ValueError):
        ModelManifest(
            name="mismatched",
            weights_filename="mismatched.keras",
            variables=(
                VariableSpec("temperature", levels=(1000, 850)),
                VariableSpec("u_component_of_wind", levels=(1000, 925)),
            ),
        )


def test_model_1702_single_level_vars_have_bare_channel_names():
    names = MODEL_1702_MANIFEST.channel_names()
    assert "surface_pressure" in names
    assert "2m_temperature" in names
    # single-level vars never get a level suffix
    assert "surface_pressure_1000" not in names


def test_served_classes_includes_dryline_but_not_background():
    assert "dryline" in SERVED_CLASSES
    assert "background" not in SERVED_CLASSES
    assert set(SERVED_CLASSES) == {"cold", "warm", "stationary", "occluded", "dryline"}
    assert set(SERVED_CLASSES).issubset(set(ALL_CLASSES))


def test_served_class_indices_align_with_all_classes_order():
    idx = THETA_E_UV_Q_MANIFEST.served_class_indices()
    names_at_idx = [THETA_E_UV_Q_MANIFEST.all_classes[i] for i in idx]
    assert names_at_idx == list(SERVED_CLASSES)


def test_spix_manifest_flat_channel_count_matches_training_config():
    # sooner_train_3d.yaml: same 4 pressure-level vars x 6 levels (24) as
    # theta-e_uv_q, plus 4 single-level vars the training pipeline broadcasts
    # across the level axis in volume mode -- flat/assembled count is still
    # 24 + 4 = 28 (the broadcast only happens at predict time, in
    # VolumetricKerasPredictor, not in the assembled grid itself).
    assert SPIX_WCE_DILATION_MANIFEST.n_channels == 28
    assert SPIX_WCE_DILATION_MANIFEST.volumetric is True
    assert THETA_E_UV_Q_MANIFEST.volumetric is False


def test_spix_manifest_variable_order_matches_training_config_for_volume_reshape():
    # VolumetricKerasPredictor's reshape assumes pressure vars come first (in
    # this exact order) and single-level vars last, matching
    # sooner_train_3d.yaml's `variables:` list -- get this wrong and the
    # model's baked-in per-(level,variable) normalization silently applies to
    # the wrong channel again (the same bug class as the 2026-08-20
    # level-major/variable-major postmortem for the flat model).
    pressure_names = [v.name for v in SPIX_WCE_DILATION_MANIFEST.pressure_level_variables()]
    single_names = [v.name for v in SPIX_WCE_DILATION_MANIFEST.single_level_variables()]
    assert pressure_names == [
        "equivalent_potential_temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
    ]
    assert single_names == [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "geopotential_at_surface",
    ]
    assert SPIX_WCE_DILATION_MANIFEST.shared_levels == (1000, 925, 850, 700, 500, 300)


def test_get_manifest_looks_up_by_name():
    assert get_manifest("theta-e_uv_q") is THETA_E_UV_Q_MANIFEST
    assert get_manifest("model_1702") is MODEL_1702_MANIFEST
    assert get_manifest("spix_wce_loss_modified_dilation") is SPIX_WCE_DILATION_MANIFEST


def test_get_manifest_unknown_model_raises():
    with pytest.raises(KeyError):
        get_manifest("not_a_real_model")


def test_variable_spec_rejects_empty_levels_tuple():
    with pytest.raises(ValueError):
        VariableSpec("temperature", levels=())


def test_variable_spec_rejects_duplicate_levels():
    with pytest.raises(ValueError):
        VariableSpec("temperature", levels=(1000, 1000))


def test_manifest_rejects_duplicate_variable_names():
    with pytest.raises(ValueError):
        ModelManifest(
            name="dup",
            weights_filename="dup.keras",
            variables=(
                VariableSpec("temperature", levels=(1000,)),
                VariableSpec("temperature", levels=(850,)),
            ),
        )


def test_manifest_rejects_empty_variables():
    with pytest.raises(ValueError):
        ModelManifest(name="empty", weights_filename="empty.keras", variables=())


def test_manifest_rejects_served_class_not_in_all_classes():
    with pytest.raises(ValueError):
        ModelManifest(
            name="bad",
            weights_filename="bad.keras",
            variables=(VariableSpec("temperature", levels=(1000,)),),
            all_classes=("background", "cold"),
            served_classes=("cold", "warm"),
        )


def test_nbs_trof_tt_is_flat_28_channels_with_9_output_classes():
    # nbs_square_dilation_trof_tt_no_inst.keras: Input batch_shape
    # [None, None, None, 28], 9-filter sup*_Conv2D output heads.
    m = NBS_SQUARE_DILATION_TROF_TT_MANIFEST
    assert m.n_channels == 28
    assert not m.volumetric
    assert m.channel_names() == SPIX_WCE_DILATION_MANIFEST.channel_names()
    assert len(m.all_classes) == 9


def test_nbs_trof_tt_class_indices_match_saved_metric_indices():
    # class_index values from the saved model's PerClassContingencyMetric
    # entries (hss_CF=1 ... hss_INST=8).
    m = NBS_SQUARE_DILATION_TROF_TT_MANIFEST
    assert m.served_classes == (
        "cold", "warm", "stationary", "occluded", "dryline", "trough", "tropical_trough",
    )
    assert m.served_class_indices() == [1, 2, 3, 4, 5, 6, 7]
    assert "instability" not in m.served_classes
    assert get_manifest("nbs_square_dilation_trof_tt_no_inst") is m
