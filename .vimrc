" Configurações básicas
set nocompatible
set encoding=utf-8
set fileencoding=utf-8
set number
set relativenumber
set mouse=a
set clipboard=unnamedplus
set tabstop=4
set shiftwidth=4
set expandtab
set smartindent
set autoindent
set cursorline
set showmatch
set hlsearch
set incsearch
set ignorecase
set smartcase
set laststatus=2
set wildmenu
set wildmode=full
set backspace=indent,eol,start

" Plugin manager (vim-plug) - instalar depois
call plug#begin('~/.vim/plugged')
Plug 'preservim/nerdtree'
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
Plug 'tpope/vim-commentary'
Plug 'tpope/vim-surround'
Plug 'airblade/vim-gitgutter'
Plug 'jiangmiao/auto-pairs'
Plug 'sheerun/vim-polyglot'
Plug 'neoclide/coc.nvim', {'branch': 'release'}
call plug#end()

" NERDTree
map <C-n> :NERDTreeToggle<CR>

" Airline
let g:airline_theme='minimalist'
let g:airline_powerline_fonts=1
let g:airline#extensions#tabline#enabled=1

" GitGutter
let g:gitgutter_enabled=1
let g:gitgutter_map_keys=0

" Auto-pairs
let g:AutoPairsFlyMode=1

" Colorscheme
syntax enable
colorscheme desert

" Remapeamentos úteis
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <leader>qq :q!<CR>
nnoremap <leader>wq :wq<CR>
