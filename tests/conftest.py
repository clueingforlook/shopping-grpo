"""Keep repository launchers distinct from ShopSimulator's scripts package."""

# Some environment tests temporarily prepend the vendored environment to
# sys.path. Bind the public launcher package before those imports occur.
import scripts  # noqa: F401
