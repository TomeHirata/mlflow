"""add budgets table

Create Date: 2025-03-07 00:00:00.000000

"""

import time

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a3f2c1b0d9e8"
down_revision = "1bd49d398cd23"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "budgets",
        sa.Column("budget_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(length=10), nullable=False, server_default="USD"),
        sa.Column("renewal_period", sa.String(length=50), nullable=False),
        sa.Column("current_spending", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("budget_id", name="budgets_pk"),
    )
    with op.batch_alter_table("budgets", schema=None) as batch_op:
        batch_op.create_index("unique_budget_name", ["name"], unique=True)


def downgrade():
    op.drop_table("budgets")
