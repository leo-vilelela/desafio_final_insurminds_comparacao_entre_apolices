# Plataforma Inteligente para Análise e Comparação de Apólices D&O

Projeto final do curso InsurMinds (I2A2): pipeline para extrair, organizar e comparar apólices de seguro Directors & Officers (D&O), mantendo referências de página e trecho para auditoria. O desenvolvimento está sendo realizado em fases; esta primeira etapa prepara configuração segura e integração centralizada com a Groq.

## Status

- Fase 0: fundação e integração inicial com a Groq.
- Fase 1: ingestão de PDFs/imagens, extração de texto página a página e cache local.
- Fase 2: segmentação de cláusulas e extração estruturada com citação por campo.
- Fase 3: persistência em SQLite/Chroma, embeddings locais e consulta RAG com fontes.
- Fase 4: comparação entre duas apólices, resumo de diferenças e exportação Markdown/PDF/JSON.
- Interface local de demonstração: upload de duas apólices, consentimento explícito antes do envio à Groq, comparação e download de relatórios.
- Amostras públicas: condições gerais D&O da Chubb e da AIG foram obtidas para teste; fontes e versões em [data/samples/SOURCES.md](data/samples/SOURCES.md). Os PDFs locais são ignorados pelo Git.
- Validação local: 28 testes automatizados passaram; a integração com Groq foi verificada com um ping e trechos das amostras públicas.
- Os documentos de demonstração deverão ser públicos e citados; não utilizar apólices de clientes.

## Requisitos

- Python 3.11 ou superior.
- Uma chave de API da Groq, configurada localmente em `.env` (nunca versionada).

## Instalação no WSL2

Na raiz do repositório:

1. Crie e ative um ambiente virtual: `python3 -m venv .venv` e `source .venv/bin/activate`.
2. Instale PyTorch CPU (evita baixar bibliotecas CUDA/NVIDIA desnecessárias): `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`.
3. Instale as demais dependências fixadas: `pip install -r requirements.txt`.
4. Copie `.env.example` para `.env` e preencha localmente `GROQ_API_KEY=<SUA_CHAVE_GROQ>`.
5. Teste a configuração sem fazer chamadas externas com `python -m unittest discover -s tests`.
6. Teste a conectividade com a API usando `python -m src.llm.groq_client --ping`.
7. Para OCR de páginas digitalizadas, instale o Tesseract no WSL2: `sudo apt install tesseract-ocr tesseract-ocr-por`.

## Interface de demonstração

Com o ambiente virtual ativo, execute `streamlit run interface/app.py`. Selecione duas apólices públicas ou sintéticas e confirme o aviso antes de iniciar: o texto das cláusulas processadas é enviado à Groq para estruturação e comparação semântica, consumindo créditos. O aplicativo salva JSON, Markdown e PDF em `data/processed/`; o OCR e os relatórios ficam locais. O primeiro uso do OCR escaneado também requer Tesseract instalado.

O teste de conexão consome uma chamada da API. Se a chave estiver ausente, o comando encerra com uma mensagem amigável sem stack trace.

## Ingestão e OCR (Fase 1)

O pipeline aceita PDF e imagens PNG, JPG/JPEG, TIFF e BMP, valida assinatura/tamanho, calcula SHA-256 e salva texto extraído por página em `data/processed/`. PDFs com texto selecionável usam extração nativa pelo PyMuPDF; páginas sem texto suficiente e imagens usam Tesseract local em português e inglês (`por+eng`). Se uma página exigir OCR e o executável ou idioma não estiver disponível, a aplicação informa como corrigir sem imprimir o conteúdo.

O cache usa o hash SHA-256 do arquivo, a versão do pipeline e as opções de OCR. Alterar idioma, limiar de texto nativo, resolução ou timeout provoca nova entrada de cache. Os resultados em `data/processed/` são ignorados pelo Git porque podem conter dados pessoais/confidenciais. O modo offline do pipeline de ingestão não precisa de chave Groq.

Execute `python -m unittest discover -s tests -v` para validar a Fase 1. Os testes criam dois PDFs sintéticos temporários; ainda é necessário selecionar e citar duas amostras públicas de apólices antes de validar documentos reais. Não foram incluídas apólices inventadas nem documentos de clientes.

Para processar documentos locais sem chamar a Groq, execute `python main.py caminho/apolice-a.pdf caminho/apolice-b.pdf` (ou `python -m src.pipeline ...`). O terminal mostra apenas metadados e o caminho do resultado; o texto por página fica em `data/processed/`.

## Segmentação e estruturação (Fase 2)

O agente de segmentação detecta títulos numerados e seções conhecidas por heurísticas, preservando as páginas e o texto-fonte em cada chunk. Títulos ambíguos podem ser classificados em lote pelo modelo rápido; o prompt manda classificar apenas os títulos, sem enviar as cláusulas inteiras.

O agente de extração envia uma cláusula por chamada ao modelo forte. A resposta JSON é validada por Pydantic e cada dado localizado exige `pagina`, `trecho_origem` literal e `confianca`. Citações de página e trecho são verificadas contra o chunk fornecido. Uma resposta inválida recebe uma tentativa corretiva; se ainda falhar, os campos daquela cláusula permanecem `nao_localizado` e o relatório registra somente metadados do erro. O cache de extração incorpora hash do documento e do chunk, versão do prompt e modelo.

Com `.env` configurado, execute `python -m src.pipeline --structure caminho/apolice.pdf`. O JSON estruturado é salvo em `data/processed/`; apenas metadados de uso são impressos. Esse modo faz chamadas à Groq; sem credenciais, use os testes offline para validar os contratos e o pipeline simulado.

Valide os agentes e schemas com `python -m unittest discover -s tests -v`. A suíte simula respostas da API, testa a tentativa corretiva, recusa citações não encontradas no trecho e verifica reutilização do cache. A validação de extração real ainda depende de chave local e amostras públicas de apólices.

## Persistência e consulta RAG (Fase 3)

Os dados estruturados e trechos com proveniência são salvos localmente em SQLite; o índice de similaridade usa ChromaDB em modo persistente local, sem iniciar servidor HTTP. As consultas vetoriais usam embeddings locais com Sentence Transformers; o ID do modelo é configurável em `EMBEDDING_MODEL` e fica registrado nos metadados da coleção. Nenhuma API de embeddings é chamada.

As cláusulas são divididas por página em chunks de até 1.200 caracteres, com sobreposição de 150 caracteres. O tamanho mantém contexto útil sem perder localização; a sobreposição reduz cortes em frases próximas a uma fronteira. Os valores estão configuráveis no `.env.example`. Chunks conservam arquivo, cláusula e página como metadados filtráveis.

Para processar, estruturar e indexar um PDF, use `python -m src.pipeline --index caminho/apolice.pdf`. Esse comando usa Groq para estruturação e baixa/carrega o modelo de embeddings local na primeira indexação. Depois, faça uma consulta com `python -m src.rag_cli "Qual é o prazo para aviso de sinistro?" --document-id SHA256_DA_APOLICE`. A resposta inclui arquivo e página; a busca pode ser limitada por hash.

O RAG envia ao modelo forte somente os trechos recuperados, não a base inteira. Se não houver resultados, responde `nao_localizado` sem chamar o LLM. Os testes simulam Chroma, embeddings e Groq para validar filtros, persistência, busca e citações sem baixar o modelo nem usar credenciais. O Chroma é usado somente como biblioteca local; não inicie nem exponha um servidor Chroma nesta demonstração.

## Comparação e relatório (Fase 4)

Para comparar duas apólices já estruturadas no SQLite, execute `python -m src.comparison_cli SHA256_APOLICE_A SHA256_APOLICE_B`. O código compara números de forma determinística: LMG maior e retenção/franquia ou prêmio menores podem ser mais favoráveis, desde que as moedas estejam identificadas e sejam compatíveis. Datas diferentes não são presumidas como vantagem. Coberturas e cláusulas textuais divergentes são avaliadas em um único lote pelo modelo forte, com somente os campos e trechos pertinentes.

As categorias são `igual`, `mais_favoravel_A`, `mais_favoravel_B`, `diferente_nao_comparavel` e `ausente_em_um`. Cada diferença traz justificativa e arquivo/página/trecho de ambos os lados; quando falta evidência, isso é representado explicitamente. O comando salva JSON, Markdown e PDF em `data/processed/`. Se a avaliação semântica falhar após a correção única, a diferença fica como não comparável. A comparação é informativa e não substitui revisão profissional das condições.

## Provedor de IA e modelos

A integração usa o SDK oficial `groq`, em vez de LangChain, para reduzir camadas no MVP e deixar explícita a única porta de acesso à API em `src/llm/groq_client.py`. A biblioteca `tenacity` aplica retry exponencial limitado a três tentativas, respeitando o cabeçalho `Retry-After` quando fornecido.

Os IDs padrão estão definidos no `.env.example`, não no código. A documentação de modelos da Groq foi consultada em 2026-09-29: `openai/gpt-oss-20b` para tarefas rápidas, `openai/gpt-oss-120b` para tarefas de maior qualidade e `qwen/qwen3.8-27b` como opção multimodal. Confirme a disponibilidade e os limites na documentação antes de uma apresentação, pois o catálogo pode mudar.

## Segurança e privacidade

- A chave é lida de `GROQ_API_KEY` com `python-dotenv`; nunca a registre em código, logs ou controle de versão.
- O `.gitignore` exclui `.env`, ambientes virtuais e arquivos de segredos do Streamlit.
- Os logs de chamadas registram somente agente, modelo, contagem de tokens e duração — nunca prompts ou conteúdo das apólices.
- Não envie documentos reais ou confidenciais para a demonstração.
- SQLite, cache OCR e Chroma armazenam textos/embeddings localmente sem criptografia de disco. Como a pasta atual do projeto pode ser sincronizada pelo OneDrive, use somente documentos públicos/sintéticos; não processe dados de clientes neste ambiente.

## Tecnologias implementadas até a Fase 4

Python, SDK oficial Groq, python-dotenv, Tenacity, Pydantic, PyMuPDF, Pillow, pytesseract, SQLite, ChromaDB e Sentence Transformers. O mecanismo Tesseract é um pré-requisito de sistema para OCR digitalizado; o modelo de embeddings é baixado e executado localmente. As versões das dependências Python estão fixadas em `requirements.txt`.

## Integrantes

A definir pelo grupo.

## Licença

Distribuído sob a licença MIT. Consulte [LICENSE](LICENSE).
