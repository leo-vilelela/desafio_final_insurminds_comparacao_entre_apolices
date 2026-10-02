"""Interface local mínima para extrair e comparar duas apólices D&O."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import streamlit as st

# `streamlit run interface/app.py` pode definir `interface/` como import root.
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import ConfigurationError

st.set_page_config(page_title="Comparador de apólices D&O", page_icon="🛡️", layout="wide")
st.title("Comparador de apólices D&O")
st.caption("Extração com evidências, comparação explicável e relatórios locais.")
st.warning(
    "Use somente documentos públicos ou sintéticos. Ao iniciar, o texto das cláusulas "
    "processadas será enviado à API Groq; não envie dados confidenciais."
)

col_a, col_b = st.columns(2)
with col_a:
    upload_a = st.file_uploader(
        "Apólice A", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"], key="policy_a"
    )
with col_b:
    upload_b = st.file_uploader(
        "Apólice B", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"], key="policy_b"
    )

consent = st.checkbox(
    "Confirmo que os arquivos são apropriados para processamento pela API Groq."
)
run = st.button("Extrair, comparar e gerar relatórios", type="primary")

if run:
    if upload_a is None or upload_b is None:
        st.error("Selecione os dois documentos antes de continuar.")
    elif not consent:
        st.error("Confirme o aviso de privacidade para iniciar o processamento.")
    else:
        try:
            from src.agents.comparison import ComparisonAgent
            from src.agents.report import ComparisonReportAgent
            from src.config import get_ingestion_settings, get_settings
            from src.llm.groq_client import get_gateway
            from src.pipeline import process_and_structure_document

            ingestion_settings = get_ingestion_settings()
            llm_settings = get_settings()
            uploads = (upload_a, upload_b)
            reports = []
            with st.spinner("Extraindo evidências e comparando as apólices…"):
                with tempfile.TemporaryDirectory(prefix="do-compare-") as temp_root:
                    for index, uploaded in enumerate(uploads):
                        if uploaded.size > ingestion_settings.max_document_mb * 1024 * 1024:
                            raise ValueError(
                                f"{uploaded.name} excede o limite de "
                                f"{ingestion_settings.max_document_mb} MB."
                            )
                        side_dir = Path(temp_root) / str(index)
                        side_dir.mkdir()
                        filename = Path(uploaded.name.replace("\\", "/")).name
                        if not filename:
                            raise ValueError("Um arquivo enviado tem nome inválido.")
                        source = side_dir / filename
                        source.write_bytes(uploaded.getvalue())
                        _, _, report, _, _ = process_and_structure_document(
                            source,
                            ingestion_settings=ingestion_settings,
                            llm_settings=llm_settings,
                        )
                        reports.append(report.model_copy(update={"source_name": filename}))

                comparison = ComparisonAgent(
                    gateway=get_gateway(llm_settings),
                    model_strong=llm_settings.model_strong,
                    processed_dir=ingestion_settings.processed_dir,
                ).compare(reports[0], reports[1])
                artifacts = ComparisonReportAgent().generate(
                    comparison, ingestion_settings.processed_dir
                )

            st.subheader("Resumo executivo")
            st.write(comparison.executive_summary)
            st.subheader("Diferenças identificadas")
            if comparison.differences:
                st.dataframe(
                    [
                        {
                            "Campo": item.label,
                            "Apólice A": item.value_a,
                            "Apólice B": item.value_b,
                            "Classificação": item.classification.value,
                            "Justificativa": item.justification,
                            "Evidência A": f"{item.citation_a.source_name}, p. {item.citation_a.page_number}",
                            "Evidência B": f"{item.citation_b.source_name}, p. {item.citation_b.page_number}",
                        }
                        for item in comparison.differences
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Nenhuma diferença nos campos localizados.")

            st.subheader("Baixar relatórios")
            download_columns = st.columns(3)
            for column, path, label, mime in zip(
                download_columns,
                (artifacts.markdown_path, artifacts.pdf_path, artifacts.json_path),
                ("Markdown", "PDF", "JSON"),
                ("text/markdown", "application/pdf", "application/json"),
                strict=True,
            ):
                column.download_button(
                    f"Baixar {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    key=f"download_{label.lower()}",
                )
            st.caption(f"Artefatos também salvos localmente em {artifacts.pdf_path.parent}.")
        except (ConfigurationError, OSError, RuntimeError, ValueError) as error:
            st.error(f"Não foi possível concluir a comparação: {error}")
