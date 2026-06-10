from abc import ABC, abstractmethod


class BaseCheck(ABC):
    """
    A generic check class meant to be inherited by concrete check implementations.
    Subclasses must register their check methods into `self.checks`.
    """

    def __init__(self, log, path):
        self.checks = []
        self.log = log
        self.path = path
        self.name = "base checks"
        pass

    def log_violation(self, rule_id, rule_name, path, msg, *args):
        """Log a Rules.md violation in the canonical locked format.

        Emits ``[<rule_id> <rule_name>] <path>: <msg>`` through
        ``self.log.error``, passing ``*args`` through for ``logging``'s lazy
        ``%``-style formatting (consistent with the ``self.log.error("%s ...",
        x)`` pattern used throughout the existing checks).

        Locked format (D-07): one space between rule_id and rule_name;
        colon and single space between path and msg.

        Args:
            rule_id: Dotted rule ID from Rules.md (e.g. ``"2.1.2"``).
            rule_name: camelCase rule name from Rules.md
                (e.g. ``"topLevelSubdirectories"``).
            path: The filesystem path where the violation was detected.
            msg: A ``%``-style format string describing the violation.
            *args: Format arguments for ``msg``.
        """
        prefix = "[%s %s] %s: " % (rule_id, rule_name, path)
        self.log.error(prefix + msg, *args)

    def warn_violation(self, rule_id, rule_name, path, msg, *args):
        """Warning-level counterpart to ``log_violation``.

        Emits ``[<rule_id> <rule_name>] <path>: <msg>`` through
        ``self.log.warning``.  Used when a condition deviates from the spec
        but is not definitively an error — for example, when STRUCT-06 runs
        against a CLOSED submission but no reference checksum is configured
        (D-12), or when the MD5 predicate encounters a symlink (D-13).

        Args:
            rule_id: Dotted rule ID from Rules.md (e.g. ``"2.1.6"``).
            rule_name: camelCase rule name from Rules.md
                (e.g. ``"codeDirectoryContents"``).
            path: The filesystem path where the condition was detected.
            msg: A ``%``-style format string describing the condition.
            *args: Format arguments for ``msg``.
        """
        prefix = "[%s %s] %s: " % (rule_id, rule_name, path)
        self.log.warning(prefix + msg, *args)

    def run_checks(self):
        """
        Execute all registered checks. Returns True if all checks pass, False otherwise.
        """
        valid = True
        errors = []
        for check in self.checks:
            try:
                v = self.execute(check)
                valid &= v
            except BaseException:
                valid &= False
                self.log.error(
                    "Exception occurred in %s while running %s in %s",
                    self.path,
                    check.__name__,
                    self.__class__.__name__)
        return valid

    def execute(self, check):
        """Custom execution of a single check method."""
        return check()

    def __call__(self):
        """Allows the check instance to be called like a function."""
        self.log.info("Starting %s for: %s", self.name, self.path)
        valid = self.run_checks()
        if valid:
            self.log.info("All %s checks passed for: %s", self.name, self.path)
        else:
            self.log.error(
                "Some %s Checks failed for: %s",
                self.name,
                self.path)
        return valid
