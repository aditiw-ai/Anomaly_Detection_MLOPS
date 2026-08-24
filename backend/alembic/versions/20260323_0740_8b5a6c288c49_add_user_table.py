"""Add user table

Revision ID: 8b5a6c288c49
Revises: d88a8ed8de51
Create Date: 2026-03-23 07:40:21.284653+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '8b5a6c288c49'
down_revision: Union[str, None] = 'd88a8ed8de51'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "users" not in tables:
        op.create_table(
            "users",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("email", sa.String(), nullable=False),
            sa.Column("hashed_password", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("roles", postgresql.ARRAY(sa.String()), nullable=False, server_default=sa.text("ARRAY['VIEWER']::varchar[]")),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.PrimaryKeyConstraint("id", name=op.f("users_pkey")),
        )
        op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
        return

    columns = {column["name"]: column for column in inspector.get_columns("users")}

    if "id" in columns and isinstance(columns["id"]["type"], postgresql.UUID):
        op.execute("ALTER TABLE users ALTER COLUMN id TYPE VARCHAR USING id::text")

    if "roles" in columns and not isinstance(columns["roles"]["type"], postgresql.ARRAY):
        op.execute(
            """
            ALTER TABLE users
            ALTER COLUMN roles TYPE VARCHAR[]
            USING CASE
                WHEN roles IS NULL THEN ARRAY['VIEWER']::VARCHAR[]
                WHEN jsonb_typeof(roles::jsonb) = 'array'
                    THEN ARRAY(SELECT jsonb_array_elements_text(roles::jsonb))
                WHEN jsonb_typeof(roles::jsonb) = 'string'
                    THEN ARRAY[trim(both '"' from roles::text)]::VARCHAR[]
                ELSE ARRAY['VIEWER']::VARCHAR[]
            END
            """
        )

    if "is_superuser" in columns:
        op.execute("ALTER TABLE users DROP COLUMN is_superuser")

    if "hashed_password" not in columns:
        op.add_column("users", sa.Column("hashed_password", sa.String(), nullable=False, server_default=""))
    if "name" not in columns:
        op.add_column("users", sa.Column("name", sa.String(), nullable=False, server_default="User"))
    if "is_active" not in columns:
        op.add_column("users", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    if "created_at" not in columns:
        op.add_column("users", sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")))
    if "updated_at" not in columns:
        op.add_column("users", sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")))

    indexes = inspector.get_indexes("users")
    has_unique_email_index = any(index.get("unique") and index.get("column_names") == ["email"] for index in indexes)
    if not has_unique_email_index:
        op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "users" in inspector.get_table_names():
        for index in inspector.get_indexes("users"):
            if index.get("name") == op.f("ix_users_email"):
                op.drop_index(op.f("ix_users_email"), table_name="users")
                break
        op.drop_table("users")
