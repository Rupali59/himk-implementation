# Multi-stage: (0) build thesis PDF, (1) Python app + copy thesis.
# Build from repo root (Thesis/):  docker build -f implementation/Dockerfile .
# So context has implementation/ and latex/.

# ---- Stage 0: build thesis ----
FROM debian:bookworm-slim AS latex
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    latexmk \
    texlive-latex-base \
    texlive-bibtex-extra \
    texlive-fonts-recommended \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*
COPY latex/ /latex/
WORKDIR /latex
RUN make pdf || true
RUN if [ -f build/Thesis.pdf ]; then cp build/Thesis.pdf /thesis.pdf; else touch /thesis.pdf; fi

# ---- Stage 1: Python app ----
FROM python:3.11-slim

WORKDIR /app

COPY implementation/requirements.txt .
RUN sed 's/opencv-python>/opencv-python-headless>/' requirements.txt > requirements.docker.txt && \
    pip install --no-cache-dir -r requirements.docker.txt

COPY implementation/ /app/
COPY --from=latex /thesis.pdf /app/thesis.pdf

ENV PORT=5050
ENV THESIS_PDF_PATH=/app/thesis.pdf
EXPOSE 5050

# Default: run full pipeline then serve (override with docker-compose command if needed)
CMD ["python", "run_all.py", "--all", "--serve"]
