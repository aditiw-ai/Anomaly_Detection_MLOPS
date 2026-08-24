from app.core.auth import Role, normalize_roles


def test_normalize_roles_handles_single_string_and_list_values():
    assert normalize_roles("ADMIN") == [Role.ADMIN]
    assert normalize_roles(["ADMIN"]) == [Role.ADMIN]
    assert normalize_roles('["ADMIN", "ML_ENGINEER"]') == [Role.ADMIN, Role.ML_ENGINEER]
    assert normalize_roles("DATA_ENGINEER,VIEWER") == [Role.DATA_ENGINEER, Role.VIEWER]


def test_admin_has_expected_permissions():
    roles = normalize_roles("ADMIN")
    permissions = set()
    for role in roles:
        permissions.update({p for p in __import__('app.core.auth', fromlist=['ROLE_PERMISSIONS']).ROLE_PERMISSIONS.get(role, [])})
    assert Role.ADMIN in roles
    assert "users:manage" in {p.value for p in permissions}
    assert "model:train" in {p.value for p in permissions}
