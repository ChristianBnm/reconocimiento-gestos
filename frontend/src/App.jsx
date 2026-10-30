
import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API = "/api";

const DATASET_ROLES = [
  { value: "train_source", label: "Entrenamiento" },
  { value: "final_test", label: "Prueba final" },
  { value: "external_eval", label: "Evaluación externa" },
  { value: "unused", label: "No utilizar" },
];

const ROLE_LABELS = {
  train_source: "Entrenamiento",
  final_test: "Prueba final",
  external_eval: "Evaluación externa",
  unused: "No utilizar",
};

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });

  const text = await response.text();
  let data = null;

  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const detail =
      typeof data?.detail === "string"
        ? data.detail
        : Array.isArray(data?.detail)
          ? data.detail
              .map((item) => item.msg ?? JSON.stringify(item))
              .join("; ")
          : `Error HTTP ${response.status}`;

    throw new Error(`${detail} (HTTP ${response.status})`);
  }

  return data;
}

function App() {
  const [health, setHealth] = useState("checking");
  const [experiments, setExperiments] = useState([]);
  const [models, setModels] = useState([]);
  const [sources, setSources] = useState([]);
  const [error, setError] = useState("");
  const [formError, setFormError] = useState("");
  const [success, setSuccess] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);

  const [form, setForm] = useState({
    name: "",
    description: "",
    model_name: "",
    source_name: "",
    role: "train_source",
  });

  const loadDashboard = useCallback(async () => {
    setError("");

    try {
      const [healthData, experimentData, modelData, sourceData] =
        await Promise.all([
          request("/health"),
          request("/experiments"),
          request("/models"),
          request("/sources"),
        ]);

      setHealth(healthData?.status === "ok" ? "online" : "error");

      setExperiments(
        Array.isArray(experimentData)
          ? experimentData
          : experimentData?.experiments ?? [],
      );

      setModels(
        Array.isArray(modelData) ? modelData : modelData?.models ?? [],
      );

      setSources(
        Array.isArray(sourceData)
          ? sourceData
          : sourceData?.sources ?? [],
      );
    } catch (err) {
      setHealth("error");
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  function updateForm(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  }

  function openForm() {
    setFormError("");
    setSuccess("");
    setForm({
      name: "",
      description: "",
      model_name: models[0]?.name ?? "",
      source_name: sources[0]?.name ?? "",
      role: "train_source",
    });
    setShowForm(true);
  }

  async function createExperiment(event) {
    event.preventDefault();
    setFormError("");
    setSuccess("");

    if (!form.name.trim()) {
      setFormError("Ingresá un nombre para el experimento.");
      return;
    }

    if (!form.model_name) {
      setFormError("Seleccioná un modelo.");
      return;
    }

    if (!form.source_name) {
      setFormError("Seleccioná una fuente de datos.");
      return;
    }

    // Esta primera versión admite una única fuente por experimento.
    // La API exige que exista al menos una fuente de entrenamiento.
    if (form.role !== "train_source") {
      setFormError(
        "En esta versión, la fuente debe tener el rol Entrenamiento.",
      );
      return;
    }

    const selectedSource = sources.find(
      (source) => source.name === form.source_name,
    );

    if (!selectedSource?.path) {
      setFormError("La fuente seleccionada no tiene una ruta válida.");
      return;
    }

    const payload = {
      name: form.name.trim(),
      description: form.description.trim(),
      model_name: form.model_name,
      datasets: [
        {
          source_name: selectedSource.name,
          path: selectedSource.path,
          role: form.role,
        },
      ],
      parameters: {},
    };

    setSaving(true);

    try {
      const created = await request("/experiments", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      setExperiments((current) => [
        created,
        ...current.filter(
          (item) => item.experiment_id !== created.experiment_id,
        ),
      ]);

      setSuccess(`Experimento creado: ${created.experiment_id}`);
      setShowForm(false);
      setForm({
        name: "",
        description: "",
        model_name: models[0]?.name ?? "",
        source_name: sources[0]?.name ?? "",
        role: "train_source",
      });
    } catch (err) {
      setFormError(err.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">LSA</div>
          <div>
            <strong>Gesture Lab</strong>
            <span>Research workspace</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav>
          <a className="nav-item active" href="#dashboard">
            ▦ Dashboard
          </a>
          <a className="nav-item" href="#experiments">
            ◈ Experimentos
          </a>
          <a className="nav-item" href="#runs">
            ▷ Ejecuciones
          </a>
          <a className="nav-item" href="#resources">
            ⌘ Recursos
          </a>
        </nav>

        <div className="sidebar-footer">
          <span className={`status-dot ${health}`} />
          API{" "}
          {health === "online"
            ? "conectada"
            : health === "checking"
              ? "verificando..."
              : "sin conexión"}
        </div>
      </aside>

      <main className="main-content" id="dashboard">
        <header className="topbar">
          <div>
            <span className="eyebrow">
              TESIS · RECONOCIMIENTO DE GESTOS
            </span>
            <h1>Panel de investigación</h1>
            <p>
              Administrá experimentos y analizá ejecuciones de modelos.
            </p>
          </div>

          <div className="api-badge">
            <span className={`status-dot ${health}`} />
            {health === "online"
              ? "Sistema operativo"
              : health === "checking"
                ? "Conectando"
                : "Error de conexión"}
          </div>
        </header>

        {error && (
          <div className="error-banner">
            No se pudieron cargar todos los datos de la API: {error}
            <button type="button" onClick={loadDashboard}>
              Reintentar
            </button>
          </div>
        )}

        {success && <div className="success-banner">{success}</div>}

        <section className="stats-grid">
          <article className="stat-card">
            <span className="stat-icon purple">◈</span>
            <span className="stat-label">Experimentos</span>
            <strong>{experiments.length}</strong>
            <span className="stat-caption">Registrados en el sistema</span>
          </article>

          <article className="stat-card">
            <span className="stat-icon blue">⌘</span>
            <span className="stat-label">Modelos disponibles</span>
            <strong>{models.length}</strong>
            <span className="stat-caption">Opciones de arquitectura</span>
          </article>

          <article className="stat-card">
            <span className="stat-icon green">▤</span>
            <span className="stat-label">Fuentes de datos</span>
            <strong>{sources.length}</strong>
            <span className="stat-caption">Fuentes configuradas</span>
          </article>

          <article className="stat-card">
            <span className="stat-icon orange">◉</span>
            <span className="stat-label">API</span>
            <strong className="api-stat">
              {health === "online"
                ? "Online"
                : health === "checking"
                  ? "..."
                  : "Error"}
            </strong>
            <span className="stat-caption">FastAPI · REST</span>
          </article>
        </section>

        <section className="content-panel" id="experiments">
          <div className="panel-heading">
            <div>
              <h2>Experimentos</h2>
              <p>Configuraciones registradas para la investigación.</p>
            </div>

            <div className="panel-actions">
              <span className="count-badge">
                {experiments.length} total
              </span>
              <button
                type="button"
                className="primary-button"
                onClick={openForm}
              >
                + Nuevo experimento
              </button>
            </div>
          </div>

          {showForm && (
            <form className="experiment-form" onSubmit={createExperiment}>
              <div className="form-heading">
                <div>
                  <h3>Crear experimento</h3>
                  <p>Definí el modelo y la fuente de datos inicial.</p>
                </div>

                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    setShowForm(false);
                    setFormError("");
                  }}
                  disabled={saving}
                >
                  Cancelar
                </button>
              </div>

              <div className="form-grid">
                <label>
                  Nombre *
                  <input
                    name="name"
                    value={form.name}
                    onChange={updateForm}
                    placeholder="Ej. ViT Tiny - Dataset original"
                    required
                    maxLength={120}
                  />
                </label>

                <label>
                  Modelo *
                  <select
                    name="model_name"
                    value={form.model_name}
                    onChange={updateForm}
                    required
                  >
                    <option value="">Seleccionar modelo</option>
                    {models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name} — {model.family}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  Fuente de datos *
                  <select
                    name="source_name"
                    value={form.source_name}
                    onChange={updateForm}
                    required
                  >
                    <option value="">Seleccionar fuente</option>
                    {sources.map((source) => (
                      <option key={source.name} value={source.name}>
                        {source.name} — {source.description}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  Rol de la fuente *
                  <select
                    name="role"
                    value={form.role}
                    onChange={updateForm}
                    required
                  >
                    {DATASET_ROLES.map((role) => (
                      <option key={role.value} value={role.value}>
                        {role.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="form-full">
                  Descripción
                  <textarea
                    name="description"
                    value={form.description}
                    onChange={updateForm}
                    placeholder="Objetivo o hipótesis del experimento"
                    rows={3}
                  />
                </label>
              </div>

              <div className="form-note">
                La ID se genera automáticamente. Para que el experimento
                pueda crearse, al menos una fuente debe tener el rol
                Entrenamiento.
              </div>

              {formError && (
                <div className="error-banner">{formError}</div>
              )}

              <div className="form-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    setShowForm(false);
                    setFormError("");
                  }}
                  disabled={saving}
                >
                  Cancelar
                </button>

                <button
                  type="submit"
                  className="primary-button"
                  disabled={
                    saving || models.length === 0 || sources.length === 0
                  }
                >
                  {saving ? "Creando..." : "Crear experimento"}
                </button>
              </div>
            </form>
          )}

          {/* Tabla de experimentos: encabezado y filas tienen seis columnas. */}
          {experiments.length > 0 ? (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Experimento</th>
                    <th>Modelo</th>
                    <th>Fuente de datos</th>
                    <th>Rol</th>
                    <th>Descripción</th>
                    <th>ID</th>
                  </tr>
                </thead>

                <tbody>
                  {experiments.map((experiment, index) => (
                    <tr key={experiment.experiment_id ?? index}>
                      <td>
                        <strong>{experiment.name ?? "Sin nombre"}</strong>
                      </td>

                      <td>{experiment.model_name ?? "—"}</td>

                      <td>
                        {experiment.datasets?.length
                          ? experiment.datasets.map((dataset, datasetIndex) => (
                              <div
                                key={`${dataset.source_name}-${datasetIndex}`}
                              >
                                {dataset.source_name ?? "—"}
                              </div>
                            ))
                          : "—"}
                      </td>

                      <td>
                        {experiment.datasets?.length
                          ? experiment.datasets.map((dataset, datasetIndex) => (
                              <div
                                key={`${dataset.source_name}-${datasetIndex}`}
                              >
                                {ROLE_LABELS[dataset.role] ??
                                  dataset.role ??
                                  "—"}
                              </div>
                            ))
                          : "—"}
                      </td>

                      <td>{experiment.description || "—"}</td>

                      <td>
                        <code>
                          {experiment.experiment_id ?? "Sin ID"}
                        </code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">◈</div>
              <h3>
                {error
                  ? "No se pudieron cargar los experimentos"
                  : "Todavía no hay experimentos"}
              </h3>
              <p>
                {error
                  ? "Revisá la conexión con la API y los registros del backend."
                  : "Creá tu primer experimento para comenzar a registrar configuraciones."}
              </p>
            </div>
          )}
        </section>

        <section className="content-panel" id="runs">
          <div className="panel-heading">
            <div>
              <h2>Ejecuciones recientes</h2>
              <p>Seguimiento de los entrenamientos realizados.</p>
            </div>
          </div>

          <div className="empty-state compact">
            <p>
              La consulta de ejecuciones y sus métricas será el próximo paso
              de integración.
            </p>
          </div>
        </section>

        <section className="content-panel" id="resources">
          <div className="panel-heading">
            <div>
              <h2>Recursos disponibles</h2>
              <p>Modelos y fuentes de datos expuestos por la API.</p>
            </div>
          </div>

          <div className="resource-grid">
            <div className="resource-card">
              <span>MODELOS</span>
              <strong>{models.length}</strong>

              {models.length > 0 ? (
                models.map((model) => (
                  <p key={model.name}>
                    {model.name}{" "}
                    <span className="resource-detail">{model.family}</span>
                  </p>
                ))
              ) : (
                <p>No se encontraron modelos.</p>
              )}
            </div>

            <div className="resource-card">
              <span>FUENTES DE DATOS</span>
              <strong>{sources.length}</strong>

              {sources.length > 0 ? (
                sources.map((source) => (
                  <p key={source.name}>
                    {source.name}
                    <span className="resource-detail">
                      {source.description}
                    </span>
                  </p>
                ))
              ) : (
                <p>No se encontraron fuentes.</p>
              )}
            </div>
          </div>
        </section>

        <footer>
          <span>Gesture Lab · Proyecto de tesis</span>
          <span>React + FastAPI + Docker</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
