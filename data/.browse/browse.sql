CREATE TABLE current_notebook_id(id UUID NOT NULL);
CREATE TABLE has_onboarded(has_onboarded BOOLEAN);
CREATE TABLE notebook_versions(
    notebook_id UUID,
    version INTEGER,
    title VARCHAR NOT NULL,
    json VARCHAR NOT NULL,
    created TIMESTAMP NOT NULL,
    expires TIMESTAMP,
    PRIMARY KEY(notebook_id, version)
);
CREATE TABLE notebooks(
    id UUID PRIMARY KEY,
    name VARCHAR NOT NULL,
    created TIMESTAMP NOT NULL
);

INSERT INTO notebooks
VALUES ('524e4147-796d-4000-8000-000000000001', 'rnagym_data', current_timestamp);

INSERT INTO notebook_versions
SELECT
    '524e4147-796d-4000-8000-000000000001',
    1,
    'RNAGym data',
    to_json({
        cells: [
            {query: 'FROM ''rnagym_sequences.parquet'';', useDatabase: 'memory', cellId: 1, isActive: true},
            {query: 'FROM ''rnagym_rfams.parquet'';', useDatabase: 'memory', cellId: 2, isActive: false},
            {query: 'FROM ''2d/rnagym_mapping.parquet'';', useDatabase: 'memory', cellId: 3, isActive: false},
            {query: 'FROM ''2d/rnagym_2d.parquet'';', useDatabase: 'memory', cellId: 4, isActive: false},
            {query: 'FROM ''3d/rnagym_3d.parquet'';', useDatabase: 'memory', cellId: 5, isActive: false},
            {query: 'FROM ''3d/rnagym_3d_scores.parquet'';', useDatabase: 'memory', cellId: 6, isActive: false}
        ],
        currentDatabase: 'memory',
        viewMode: {mode: 'default'},
        version: 1,
        notebookSerializationFormat: 3
    })::VARCHAR,
    current_timestamp,
    NULL;

INSERT INTO current_notebook_id
VALUES ('524e4147-796d-4000-8000-000000000001');

INSERT INTO has_onboarded VALUES (true);
