"""Privacy consent, authoritative-source permission, knowledge graph and agent-run tables.

Revision ID: 012
Revises: 011
Create Date: 2026-09-28

Adds the six tables that three previously-unregistered feature layers depend on:

* ``data_consents``        - DPDP notice + consent records (app/routers/privacy.py)
* ``source_consents``      - explicit, logged permission for a paid/third-party
                             source connector (the "user's own paid subscriptions
                             only with explicit, logged permission" requirement)
* ``source_access_logs``   - the evidence trail for that permission
* ``graph_nodes``          - relational knowledge graph entities
* ``graph_edges``          - relational knowledge graph relationships
* ``agent_runs``           - persisted agentic-orchestration traces

The DDL below was generated directly from the SQLAlchemy models
(``scripts/_gen_migration_012.py``) so it cannot drift from the code.

Design notes
------------
* ``DataConsent``/``SourceConsent`` cascade with the account because they are
  personal data. ``SourceAccessLog.user_id`` and ``.source_consent_id`` are
  SET NULL so the permission trail survives erasure without a personal
  identifier - mirroring how ``AuditLog`` behaves.
* ``graph_edges.source_document_id`` is SET NULL (not CASCADE) so deleting a
  corpus document never silently rewrites graph history.
* Everything is additive: nothing existing is dropped or altered, which is why
  ``downgrade()`` only drops these six tables.

An earlier ``alembic revision --autogenerate`` run against the live database
proposed additionally dropping ``roles_name_key``, dropping a ``Clients`` table
and rebuilding several GIN/unique indexes. Those were rejected as unrelated
drift between the legacy migrations and the models and are deliberately NOT
part of this revision.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TABLES = (
    "data_consents",
    "source_consents",
    "source_access_logs",
    "graph_nodes",
    "graph_edges",
    "agent_runs",
)


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE data_consents (
            id SERIAL NOT NULL,
            user_id INTEGER NOT NULL,
            notice_version VARCHAR(50) NOT NULL,
            purposes TEXT NOT NULL,
            granted BOOLEAN NOT NULL,
            granted_at TIMESTAMP WITH TIME ZONE,
            withdrawn_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """
    )
    op.create_index("ix_data_consents_id", "data_consents", ["id"], unique=False)
    op.create_index("ix_data_consents_user_id", "data_consents", ["user_id"], unique=False)

    op.execute(
        """
        CREATE TABLE source_consents (
            id SERIAL NOT NULL,
            user_id INTEGER NOT NULL,
            source_id VARCHAR(100) NOT NULL,
            source_title VARCHAR(255) NOT NULL,
            access_type VARCHAR(50) NOT NULL,
            scope TEXT NOT NULL,
            permission VARCHAR(20) NOT NULL,
            granted_at TIMESTAMP WITH TIME ZONE,
            revoked_at TIMESTAMP WITH TIME ZONE,
            expires_at TIMESTAMP WITH TIME ZONE,
            ip_address VARCHAR(45),
            user_agent TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """
    )
    op.create_index("ix_source_consents_id", "source_consents", ["id"], unique=False)
    op.create_index("ix_source_consents_user_id", "source_consents", ["user_id"], unique=False)
    op.create_index("ix_source_consents_source_id", "source_consents", ["source_id"], unique=False)

    op.execute(
        """
        CREATE TABLE source_access_logs (
            id SERIAL NOT NULL,
            user_id INTEGER,
            source_consent_id INTEGER,
            source_id VARCHAR(100) NOT NULL,
            action VARCHAR(50) NOT NULL,
            detail TEXT,
            ip_address VARCHAR(45),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE SET NULL,
            FOREIGN KEY(source_consent_id) REFERENCES source_consents (id) ON DELETE SET NULL
        )
        """
    )
    op.create_index("ix_source_access_logs_id", "source_access_logs", ["id"], unique=False)
    op.create_index("ix_source_access_logs_user_id", "source_access_logs", ["user_id"], unique=False)
    op.create_index("ix_source_access_logs_source_id", "source_access_logs", ["source_id"], unique=False)
    op.create_index("ix_source_access_logs_action", "source_access_logs", ["action"], unique=False)

    op.execute(
        """
        CREATE TABLE graph_nodes (
            id SERIAL NOT NULL,
            node_type VARCHAR(40) NOT NULL,
            canonical_name VARCHAR(500) NOT NULL,
            normalized_name VARCHAR(500) NOT NULL,
            attributes_json TEXT,
            source_document_ids TEXT,
            jurisdiction VARCHAR(100),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            CONSTRAINT uq_graph_nodes_normalized_name UNIQUE (normalized_name)
        )
        """
    )
    op.create_index("ix_graph_nodes_id", "graph_nodes", ["id"], unique=False)
    op.create_index("ix_graph_nodes_node_type", "graph_nodes", ["node_type"], unique=False)
    op.create_index("ix_graph_nodes_normalized_name", "graph_nodes", ["normalized_name"], unique=True)
    op.create_index("ix_graph_nodes_jurisdiction", "graph_nodes", ["jurisdiction"], unique=False)
    op.create_index("ix_graph_nodes_type_name", "graph_nodes", ["node_type", "normalized_name"], unique=False)

    op.execute(
        """
        CREATE TABLE graph_edges (
            id SERIAL NOT NULL,
            from_node_id INTEGER NOT NULL,
            to_node_id INTEGER NOT NULL,
            relation VARCHAR(60) NOT NULL,
            weight FLOAT NOT NULL,
            evidence_json TEXT,
            source_document_id INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            CONSTRAINT uq_graph_edges_triple UNIQUE (from_node_id, to_node_id, relation),
            FOREIGN KEY(from_node_id) REFERENCES graph_nodes (id) ON DELETE CASCADE,
            FOREIGN KEY(to_node_id) REFERENCES graph_nodes (id) ON DELETE CASCADE,
            FOREIGN KEY(source_document_id) REFERENCES source_documents (id) ON DELETE SET NULL
        )
        """
    )
    op.create_index("ix_graph_edges_id", "graph_edges", ["id"], unique=False)
    op.create_index("ix_graph_edges_from_node_id", "graph_edges", ["from_node_id"], unique=False)
    op.create_index("ix_graph_edges_to_node_id", "graph_edges", ["to_node_id"], unique=False)
    op.create_index("ix_graph_edges_relation", "graph_edges", ["relation"], unique=False)
    op.create_index("ix_graph_edges_from_relation", "graph_edges", ["from_node_id", "relation"], unique=False)
    op.create_index("ix_graph_edges_to_relation", "graph_edges", ["to_node_id", "relation"], unique=False)

    op.execute(
        """
        CREATE TABLE agent_runs (
            id SERIAL NOT NULL,
            user_id INTEGER NOT NULL,
            product_version_id INTEGER,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            plan_json TEXT,
            trace_json TEXT,
            citations_json TEXT,
            sources_json TEXT,
            graph_paths_json TEXT,
            warnings_json TEXT,
            confidence VARCHAR(10) NOT NULL,
            confidence_reason TEXT,
            insufficient_evidence BOOLEAN NOT NULL,
            review_required BOOLEAN NOT NULL,
            steps_used INTEGER NOT NULL,
            max_steps INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY(product_version_id) REFERENCES product_versions (id) ON DELETE SET NULL
        )
        """
    )
    op.create_index("ix_agent_runs_id", "agent_runs", ["id"], unique=False)
    op.create_index("ix_agent_runs_user_id", "agent_runs", ["user_id"], unique=False)
    op.create_index("ix_agent_runs_product_version_id", "agent_runs", ["product_version_id"], unique=False)


def downgrade() -> None:
    # Reverse dependency order: edges before nodes, logs before consents.
    op.drop_index("ix_agent_runs_product_version_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_user_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_id", table_name="agent_runs")
    op.drop_table("agent_runs")

    op.drop_index("ix_graph_edges_to_relation", table_name="graph_edges")
    op.drop_index("ix_graph_edges_from_relation", table_name="graph_edges")
    op.drop_index("ix_graph_edges_relation", table_name="graph_edges")
    op.drop_index("ix_graph_edges_to_node_id", table_name="graph_edges")
    op.drop_index("ix_graph_edges_from_node_id", table_name="graph_edges")
    op.drop_index("ix_graph_edges_id", table_name="graph_edges")
    op.drop_table("graph_edges")

    op.drop_index("ix_graph_nodes_type_name", table_name="graph_nodes")
    op.drop_index("ix_graph_nodes_jurisdiction", table_name="graph_nodes")
    op.drop_index("ix_graph_nodes_normalized_name", table_name="graph_nodes")
    op.drop_index("ix_graph_nodes_node_type", table_name="graph_nodes")
    op.drop_index("ix_graph_nodes_id", table_name="graph_nodes")
    op.drop_table("graph_nodes")

    op.drop_index("ix_source_access_logs_action", table_name="source_access_logs")
    op.drop_index("ix_source_access_logs_source_id", table_name="source_access_logs")
    op.drop_index("ix_source_access_logs_user_id", table_name="source_access_logs")
    op.drop_index("ix_source_access_logs_id", table_name="source_access_logs")
    op.drop_table("source_access_logs")

    op.drop_index("ix_source_consents_source_id", table_name="source_consents")
    op.drop_index("ix_source_consents_user_id", table_name="source_consents")
    op.drop_index("ix_source_consents_id", table_name="source_consents")
    op.drop_table("source_consents")

    op.drop_index("ix_data_consents_user_id", table_name="data_consents")
    op.drop_index("ix_data_consents_id", table_name="data_consents")
    op.drop_table("data_consents")
