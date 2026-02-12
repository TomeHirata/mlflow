"""add_webhook_permissions

Revision ID: b3c4d5e6f7a8
Revises: a1b2c3d4e5f6
Create Date: 2026-02-12 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "b3c4d5e6f7a8"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "webhook_permissions",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("webhook_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_webhook_perm_user_id"),
        sa.UniqueConstraint("webhook_id", "user_id", name="unique_webhook_user"),
    )


def downgrade() -> None:
    op.drop_table("webhook_permissions")
