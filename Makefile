CC = gcc
CFLAGS = -O2 -Wall -std=c99 -I src -lm
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files (in src/)
SRCS = $(SRCDIR)/utils.c $(SRCDIR)/activations.c $(SRCDIR)/layer.c $(SRCDIR)/network.c $(SRCDIR)/data.c $(SRCDIR)/optimizer.c
OBJS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

# Ensure output directories exist
$(shell mkdir -p $(OBJDIR) $(BINDIR))

all: xor spirals mnist

xor: $(OBJS) $(SRCDIR)/main_xor.c
	$(CC) $(CFLAGS) -o $(BINDIR)/xor.exe $(OBJS) $(SRCDIR)/main_xor.c

spirals: $(OBJS) $(SRCDIR)/main_spirals.c
	$(CC) $(CFLAGS) -o $(BINDIR)/spirals.exe $(OBJS) $(SRCDIR)/main_spirals.c

mnist: $(OBJS) $(SRCDIR)/main_mnist.c
	$(CC) $(CFLAGS) -o $(BINDIR)/mnist.exe $(OBJS) $(SRCDIR)/main_mnist.c

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/*.exe