FROM python:3.11-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=scd
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Configurar PATH
ENV PATH=/root/.local/bin:/home/$USERNAME/.local/bin:$PATH

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Instalar pipx
RUN python3 -m pip install pipx \
    && python3 -m pipx ensurepath --force 

# Criar o usuário
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Configurar diretório de trabalho
WORKDIR /app
RUN chown ${USERNAME}:${USERNAME} /app

USER $USERNAME

# Instalar Poetry
RUN pipx install poetry \
    && poetry config virtualenvs.in-project true \
    && poetry config virtualenvs.prompt "venv"

# Copiar arquivos do Poetry com as permissões corretas
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml poetry.lock ./

# Instalar dependências
RUN poetry install --no-root --no-interaction

# Copiar o resto do código
COPY --chown=${USERNAME}:${USERNAME} . .

# Instalar o projeto
RUN poetry install --no-interaction

# Executar testes por padrão
CMD ["poetry", "run", "pytest", "tests/"]